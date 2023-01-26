from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy import interpolate
from torch.utils.data import Dataset

from src.config import Origin, data_config, net_config
from src.data_utils.transforms import (
    define_augmentations,
    define_transform,
    gamma_torch,
    normalize,
)


def array_inpainting(array: np.ndarray) -> np.ndarray:
    inpainted = deepcopy(array)
    for channel in range(array.shape[-1]):
        img = array[..., channel]
        if np.isnan(img).any() or np.isinf(img).any():
            valid_mask = ~(np.isnan(img) | np.isinf(img))
            coords = np.array(np.nonzero(valid_mask)).T
            values = img[valid_mask]
            try:
                it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
                filled = it(list(np.ndindex(img.shape))).reshape(img.shape)
                inpainted[..., channel] = filled
            except Exception:
                img[np.isnan(img)] = 0.0
                img[np.isinf(img)] = 0.0
                inpainted[..., channel] = img

    return inpainted


def read_dataframe(
    data_dir: Path,
    csv_path: Path | pd.DataFrame,
    phase: str | None = None,
    test_size: int = 0,
) -> (pd.DataFrame, pd.DataFrame):
    images = data_dir.rglob("*.npz")

    images_dict = defaultdict()
    origin = defaultdict()
    for image in images:
        filename_info = str(image.stem).split("_")
        origin[filename_info[0]] = filename_info[1]
        images_dict[filename_info[0]] = image

    df_full = pd.read_csv(csv_path) if isinstance(csv_path, Path) else csv_path
    df_split = df_full if phase is None else df_full[df_full["split"] == phase]
    data = df_split if test_size <= 0 else df_split.iloc[:test_size]
    try:
        data["filepath"] = data.loc[:, "uid"].map(images_dict)
        data["origin"] = data.loc[:, "uid"].map(origin)
    except KeyError as e:
        logger.warning(f"Not all data were downloaded:\n{e}")
        data["filepath"] = data.loc[:, "uid"].apply(
            lambda x: images_dict[x] if x in images_dict.keys() else None
        )
        data["origin"] = data.loc[:, "uid"].apply(
            lambda x: origin[x] if x in origin.keys() else None
        )

    return data, df_full


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
    ):
        self.data_dir = data_dir
        self.inference = inference
        self.inpaint = inpaint
        self.hrrr = hrrr

        self.save_preprocessed = save_preprocessed
        logger.warning(
            f"Preprocessed data will be saved to or read from {self.save_preprocessed}"
        )

        self.data, self.df_full = read_dataframe(data_dir, csv_path, phase, test_size)
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

            image_orig = array[..., :3]
            meta_channels = array[..., 4:]

            image_orig = np.concatenate([image_orig, meta_channels], axis=-1).astype(
                "float32"
            )

            if self.inpaint and (
                np.isnan(image_orig).any() or np.isinf(image_orig).any()
            ):
                image_orig = array_inpainting(image_orig)

            image = normalize(image_orig, mean, std).astype("float32")

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

        sample = {
            "uid": uid,
            "image": image,
            "hrrr": [] if hrrr is None else torch.tensor(hrrr, dtype=torch.float32),
            "label": torch.tensor(label_scaled, dtype=torch.float32),
            "label_origin": label,
            "filepath": filepath,
            "severity": severity,
            "region": region,
            "origin": origin,
        }

        return sample

    def __len__(self) -> int:
        return len(self.data)
