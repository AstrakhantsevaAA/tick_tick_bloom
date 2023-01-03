from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from albumentations import (
    Affine,
    Blur,
    CoarseDropout,
    Compose,
    Downscale,
    GridDistortion,
    Perspective,
    ShiftScaleRotate,
    Resize,
)
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class AlgalDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image tensors, and labels.
    """

    def __init__(
        self,
        data_dir: Path,
        csv_path: Path,
        phase: str,
        augmentations_intensity: float = 0.0,
        test_size: int = 0,
    ):
        self.data_dir = data_dir
        self.images = data_dir.rglob("*.npy")
        self.images_dict = defaultdict()
        for image in self.images:
            self.images_dict[image.stem] = image
        df = pd.read_csv(csv_path)
        df = df[df["split"] == phase]
        self.data = df if test_size <= 0 else df.iloc[:test_size]
        self.data["filepath"] = self.data.loc[:, "uid"].apply(lambda x: self.images_dict[x])
        self.labels = self.data.loc[:, "severity"]

        self.transform = Compose(
            [
                Resize(224, 224),
                ToTensorV2(),
            ]
        )
        self.augmentation = None
        if augmentations_intensity > 0:
            self.augmentation = Compose(
                [
                    Resize(224, 224),
                    CoarseDropout(
                        p=0.99,
                    ),
                    Blur(p=0.1),
                    ShiftScaleRotate(p=1.0),
                    Affine(),
                    GridDistortion(),
                    Downscale(
                        p=0.7,
                    ),
                    Perspective(p=0.9),
                ],
                p=augmentations_intensity,
            )

    def __getitem__(self, index):
        filepath = str(self.data["filepath"].iloc[index])
        with open(filepath, "rb") as f:
            image = np.load(f)

        if image is None:
            raise Exception(
                f"image is None, got filepath: {filepath} \n data: {self.data}"
            )

        image = image.astype('float32') / 255.
        image = np.transpose(image, (1, 2, 0))

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        image = self.transform(image=image)["image"]
        label = self.data["severity"].iloc[index]
        label = int(label) - 1
        label = torch.tensor(label,  dtype=torch.long)

        sample = {
            "image": image,
            "label": label,
            "filepath": filepath,
        }

        return sample

    def __len__(self):
        return len(self.data)
