import numpy as np
import pytest
import torch
from einops import asnumpy

from src.data_utils.transforms import gamma_torch, phi_torch


def phi(x: np.ndarray) -> np.ndarray:
    x = np.where(x < 1, 0.5, x)
    x = np.where((1e1 <= x) & (x < 2e4), 0.5 * x, x)
    x = np.where((2e4 <= x) & (x < 1e5), 1.125 * x - 12500, x)
    return x


def gamma(x: np.ndarray) -> np.ndarray:
    return np.log10(phi(x) / 100).clip(1, 5)


@pytest.fixture
def example():
    return [90000, 2, 0, 14000, 2000, 10000000]


@pytest.fixture
def result():
    return [2.9481683, 1.0, 1.0, 1.845098, 1.0, 5.0]


def test_gamma(example, result):
    print(gamma_torch(torch.tensor(example)))
    assert np.allclose(
        asnumpy(gamma_torch(torch.tensor(example))), gamma(np.array(example))
    )
    assert np.allclose(asnumpy(gamma_torch(torch.tensor(example))), np.array(result))


def test_phi(example):
    assert np.allclose(
        asnumpy(phi_torch(torch.tensor(example))), phi(np.array(example))
    ), f"{asnumpy(phi_torch(torch.tensor(example)))}, {phi(np.array(example))}"
