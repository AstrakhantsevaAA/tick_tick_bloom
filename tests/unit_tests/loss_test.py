import pytest
import numpy as np
import torch
from einops import asnumpy

from src.train.classificator.loss import gamma, gamma_torch, phi, phi_torch


@pytest.fixture
def example():
    return [90000, 2, 0, 14000, 2000, 10000000]


def test_gamma(example):
    assert np.allclose(
        asnumpy(gamma_torch(torch.tensor(example))), gamma(np.array(example))
    )


def test_phi(example):
    assert np.allclose(
        asnumpy(phi_torch(torch.tensor(example))), phi(np.array(example))
    ), f"{asnumpy(phi_torch(torch.tensor(example)))}, {phi(np.array(example))}"
