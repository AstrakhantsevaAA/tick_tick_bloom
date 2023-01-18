from pathlib import Path

import pytest

from src.config import system_config
from src.data_utils.dataset import AlgalDataset


@pytest.fixture(scope="class")
def data_dir() -> Path:
    return system_config.data_dir / "arrays/more_arrays_fixed"


@pytest.fixture(scope="class")
def csv_path() -> Path:
    return system_config.data_dir / "splits/downloaded.csv"


@pytest.fixture()
def dataset(data_dir, csv_path) -> AlgalDataset:
    dataset = AlgalDataset(
        data_dir,
        csv_path,
        phase="train",
    )
    return dataset
