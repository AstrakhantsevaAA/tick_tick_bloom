from pathlib import Path

import pytest

from src.config import system_config


@pytest.fixture(scope="class")
def data_dir() -> Path:
    return system_config.data_dir / "benchmark/image_arrays"


@pytest.fixture(scope="class")
def csv_path() -> Path:
    return system_config.data_dir / "benchmark/uid_train.csv"
