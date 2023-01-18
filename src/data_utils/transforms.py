import numpy as np
from albumentations import (
    Affine,
    Blur,
    CoarseDropout,
    Compose,
    Downscale,
    Flip,
    GridDistortion,
    Perspective,
    RandomBrightnessContrast,
    RandomCrop,
    Resize,
    ShiftScaleRotate,
)
from albumentations.pytorch.transforms import ToTensorV2
from torch import Tensor, log10, manual_seed, uint8

manual_seed(17)


def phi_torch(x: Tensor) -> Tensor:
    mask = x < 1
    x = mask * 0.5 + x * (1 - mask.type(uint8))
    mask = (1e1 <= x) * (x < 2e4)
    x = x * (0.5 * mask + (1 - mask.type(uint8)))
    mask = (2e4 <= x) * (x < 1e5)
    x = ((1.125 * x - 12500) * mask) + x * (1 - mask.type(uint8))

    return x


def gamma_torch(x: Tensor) -> Tensor:
    return log10(phi_torch(x) / 100).clip(1, 5)


def define_augmentations(augmentations_intensity: float = 0.0) -> Compose:
    return Compose(
        [
            Resize(224, 224),
            CoarseDropout(
                p=0.5,
            ),
            Blur(p=0.5),
            ShiftScaleRotate(p=0.5),
            Affine(),
            GridDistortion(),
            Downscale(
                p=0.5,
            ),
            Perspective(p=0.5),
            Flip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            # ColorJitter(),
        ],
        p=augmentations_intensity,
    )


def define_transform() -> Compose:
    return Compose(
        [
            Resize(256, 256),
            RandomCrop(224, 224),
            ToTensorV2(),
        ]
    )


def normalize(input: np.ndarray, mean: list, std: list) -> np.ndarray:
    # input shape: HxWxC
    output = (input - mean) / std
    return output
