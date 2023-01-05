from torch.nn.functional import mse_loss
import numpy as np
import torch


def phi(x):
    x[x < 1] = 0.5
    x[(1e1 <= x) & (x < 2e4)] = 0.5 * x[(1e1 <= x) & (x < 2e4)]
    x[(2e4 <= x) & (x < 1e5)] = 1.25 * x[(2e4 <= x) & (x < 1e5)] - 12500
    return x


def gamma(x):
    return np.log10(phi(x) / 1000).clip(0, 5)


def phi_torch(x: torch.Tensor) -> torch.Tensor:
    mask = x < 1
    x = mask * 0.5 + x * (1 - mask.type(torch.uint8))
    mask = (1e1 <= x) * (x < 2e4)
    x = x * (0.5 * mask + (1 - mask.type(torch.uint8)))
    mask = (2e4 <= x) * (x < 1e5)
    x = ((1.25 * x - 12500) * mask) + x * (1 - mask.type(torch.uint8))

    return x


def gamma_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.log10(phi_torch(x) / 100).clip(1, 6)


def density_mse_loss(inputs: torch.Tensor, targets: torch.Tensor):
    # targets will be preprocessed in advance
    return mse_loss(gamma_torch(inputs), targets)


if __name__ == "__main__":
    print(phi_torch(torch.tensor([90000, 2, 0, 14000, 2000, 10000000])))
    print(gamma_torch(torch.tensor([90000, 2, 0, 14000, 2000, 100000000000])))
    # Output:
    # tensor([1.0000e+05, 2.0000e+00, 5.0000e-01, 7.0000e+03, 1.0000e+03, 1.0000e+07])
    # tensor([3.0000, 1.0000, 1.0000, 1.8451, 1.0000, 6.0000])






