from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from src.config import Origin, data_config, net_config
from src.data_utils import dataset_utils
from src.data_utils.transforms import (
    define_augmentations,
    define_transform,
    gamma_torch,
    normalize,
)


class AlgalDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image tensors, and labels.
    """

    def __init__(
        self,
        data_dir: Path,
        csv_path: Path | pd.DataFrame,
        phase: str | None = None,
        augmentations_intensity: float = 0.0,
        test_size: int = 0,
        inference: bool = False,
        save_preprocessed: str | Path | None = None,
        inpaint: bool = False,
        hrrr: bool = False,
        meta_channels_path: Path | str | None = None,
    ):
        self.data_dir = data_dir
        self.inference = inference
        self.inpaint = inpaint
        self.hrrr = hrrr
        self.meta_channels_path = meta_channels_path

        self.save_preprocessed = save_preprocessed
        logger.warning(
            f"Preprocessed data will be saved to or read from {self.save_preprocessed}"
        )

        self.data, self.df_full = dataset_utils.read_dataframe(
            data_dir, csv_path, phase, test_size
        )
        self.regions = self.data.loc[:, "region"]
        self.transform = define_transform()
        self.augmentation = None
        if augmentations_intensity > 0:
            self.augmentation = define_augmentations(augmentations_intensity)

    def __getitem__(self, index):
        row = self.data.iloc[index, :]
        filepath = str(row["filepath"])
        uid = str(row["uid"])
        split = str(row["split"])
        region = str(row["region"])
        origin = str(row["origin"])
        self.inference = True if split == "test" else self.inference
        severity = int(row["severity"]) if not self.inference else 0

        hrrr = None
        if self.hrrr:
            hrrr = row[data_config.best_features]
            if not hrrr.empty:
                hrrr = hrrr.to_list()
            else:
                logger.error("hrrr is empty!")

        mean = data_config.mean[Origin[origin]]
        std = data_config.std[Origin[origin]]

        save_preprocessed = (
            f"{str(self.save_preprocessed)}/{uid}.npy"
            if self.save_preprocessed is not None
            else None
        )

        if save_preprocessed is not None and Path(save_preprocessed).exists():
            with open(save_preprocessed, "rb") as f:
                image = np.load(f)
                label_scaled = np.load(f)
                label = np.load(f)

            image = image.astype("float32")
            label_scaled = label_scaled.astype("float32")
        else:
            with np.load(filepath, "r+") as f:
                array = f["caption"]

            if array is None:
                raise Exception(
                    f"image is None, got filepath: {filepath} \n data: {self.data}"
                )

            if self.inpaint and (np.isnan(array).any() or np.isinf(array).any()):
                array = dataset_utils.array_inpainting(array)

            image = normalize(array, mean, std).astype("float32")

            image_orig = image[..., :3]
            meta_channels = image[..., 4:]
            scl = image[..., 3]
            scl_channels = np.zeros((image.shape[0], image.shape[1], 2))
            if origin == Origin.sentinel:
                scl_channels[..., 0] = scl
            else:
                scl_channels[..., 1] = scl

            image = np.concatenate(
                [image_orig, meta_channels, scl_channels], axis=-1
            ).astype("float32")

            if self.meta_channels_path is not None:
                image = dataset_utils.add_meta_channels(
                    Path(self.meta_channels_path), image, uid
                ).astype("float32")

            label_scaled, label = 0.0, 0.0
            if not self.inference:
                label = self.data[net_config.label_column].iloc[index]
                if net_config.label_column == "severity":
                    label_scaled = int(label) - 1
                else:
                    label_scaled = gamma_torch(
                        torch.tensor(int(label), dtype=torch.long)
                    )

            if save_preprocessed is not None:
                Path(self.save_preprocessed).mkdir(parents=True, exist_ok=True)
                with open(save_preprocessed, "wb") as f:
                    np.save(f, image)
                    np.save(f, label_scaled)
                    np.save(f, label)

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        image = self.transform(image=image)["image"]

        if isinstance(label_scaled, (np.ndarray, float)):
            label_scaled = torch.tensor(label_scaled, dtype=torch.float32)
        else:
            label_scaled = label_scaled.type("torch.FloatTensor")

        sample = {
            "uid": uid,
            "image": image,
            "hrrr": [] if hrrr is None else torch.tensor(hrrr, dtype=torch.float32),
            "label": label_scaled,
            "label_origin": label,
            "filepath": filepath,
            "severity": severity,
            "region": region,
            "origin": origin,
        }

        return sample

    def __len__(self) -> int:
        return len(self.data)
