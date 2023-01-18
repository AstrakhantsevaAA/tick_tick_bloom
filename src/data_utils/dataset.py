from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from src.config import Origin, net_config, statistics, system_config
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
        phase: str = "train",
        augmentations_intensity: float = 0.0,
        test_size: int = 0,
        inference: bool = False,
        save_preprocessed: str | Path = system_config.data_dir / "preprocessed/test",
    ):
        self.data_dir = data_dir
        self.inference = inference
        self.images = data_dir.rglob("*.npz")
        self.images_dict = defaultdict()
        self.origin = defaultdict()
        self.save_preprocessed = (
            save_preprocessed
            if len(save_preprocessed) > 0
            else system_config.data_dir / "preprocessed/test"
        )

        for image in self.images:
            filename_info = str(image.stem).split("_")
            self.origin[filename_info[0]] = filename_info[1]
            self.images_dict[filename_info[0]] = image

        self.df_full = pd.read_csv(csv_path) if isinstance(csv_path, Path) else csv_path
        self.df_split = self.df_full[self.df_full["split"] == phase]
        self.data = self.df_split if test_size <= 0 else self.df_split.iloc[:test_size]
        try:
            self.data["filepath"] = self.data.loc[:, "uid"].map(self.images_dict)
            self.data["origin"] = self.data.loc[:, "uid"].map(self.origin)
        except KeyError as e:
            logger.warning(f"Not all data were downloaded:\n{e}")
            self.data["filepath"] = self.data.loc[:, "uid"].apply(
                lambda x: self.images_dict[x] if x in self.images_dict.keys() else None
            )
            self.data["origin"] = self.data.loc[:, "uid"].apply(
                lambda x: self.origin[x] if x in self.origin.keys() else None
            )

        self.regions = self.data.loc[:, "region"]
        self.transform = define_transform()
        self.augmentation = None
        if augmentations_intensity > 0:
            self.augmentation = define_augmentations(augmentations_intensity)

    def __getitem__(self, index):
        filepath = str(self.data["filepath"].iloc[index])
        uid = str(self.data["uid"].iloc[index])
        severity = int(self.data["severity"].iloc[index]) if not self.inference else 0
        region = str(self.data["region"].iloc[index])
        origin = str(self.data["origin"].iloc[index])

        mean = statistics.mean[Origin[origin]]
        std = statistics.std[Origin[origin]]

        save_preprocessed = f"{self.save_preprocessed}/{uid}.npz"

        if Path(save_preprocessed).exists():
            with np.load(save_preprocessed, "r+") as f:
                image = f["image"]
                label_scaled = f["label_scaled"]
                label = f["label"]

            image = image.astype("float32")
            label_scaled = label_scaled.astype("float32")

        else:
            with np.load(filepath, "r+") as f:
                array = f["caption"]

            if array is None:
                raise Exception(
                    f"image is None, got filepath: {filepath} \n data: {self.data}"
                )

            array[np.isnan(array)] = 0.0
            array[np.isinf(array)] = 0.0

            image_orig = array[..., :3]
            meta_channels = array[..., 4:]

            image = np.concatenate([image_orig, meta_channels], axis=-1)
            image = normalize(image, mean, std).astype("float32")

            label_scaled, label = 0.0, 0.0

            if not self.inference:
                label = self.data[net_config.label_column].iloc[index]
                if net_config.label_column == "severity":
                    label_scaled = int(label) - 1
                else:
                    label_scaled = gamma_torch(
                        torch.tensor(int(label), dtype=torch.long)
                    )

            Path(self.save_preprocessed).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_preprocessed, image=image, label_scaled=label_scaled, label=label
            )

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        image = self.transform(image=image)["image"]
        label_scaled = torch.tensor(label_scaled, dtype=torch.float32)

        sample = {
            "uid": uid,
            "image": image,
            # "image_original": image_orig,
            # "meta": meta_channels,
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
