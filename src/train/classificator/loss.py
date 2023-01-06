from torch.nn.functional import mse_loss
import numpy as np
import torch


def phi(x: np.ndarray) -> np.ndarray:
    x = np.where(x < 1, 0.5, x)
    x = np.where((1e1 <= x) & (x < 2e4), 0.5 * x, x)
    x = np.where((2e4 <= x) & (x < 1e5), 1.25 * x - 12500, x)
    return x


def gamma(x: np.ndarray) -> np.ndarray:
    return np.log10(phi(x) / 100).clip(1, 5)


def phi_torch(x: torch.Tensor) -> torch.Tensor:
    mask = x < 1
    x = mask * 0.5 + x * (1 - mask.type(torch.uint8))
    mask = (1e1 <= x) * (x < 2e4)
    x = x * (0.5 * mask + (1 - mask.type(torch.uint8)))
    mask = (2e4 <= x) * (x < 1e5)
    x = ((1.25 * x - 12500) * mask) + x * (1 - mask.type(torch.uint8))

    return x


def gamma_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log10(phi_torch(x) / 100).clip(1, 5)


def density_mse_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # targets will be preprocessed in advance
    return mse_loss(gamma_torch(inputs), targets)
