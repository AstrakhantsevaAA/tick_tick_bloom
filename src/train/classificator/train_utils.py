import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils.dataset import AlgalDataset
from src.config import Phase, system_config


def fix_seeds(random_state: int = 42):
    random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(
    data_dir: Path = system_config.data_dir,
    csv_path: Path | pd.DataFrame = None,
    augmentations_intensity: float = 0,
    batch_size: int = 32,
    test_size: int = 0,
    inference: bool = False,
):
    dataloader = defaultdict()

    if csv_path is None or len(csv_path) == 0:
        raise Exception(
            "csv files with train and validation data are None, for training those files are necessary"
        )

    shuffle = True
    for phase in Phase:
        if inference and phase == Phase.train:
            continue
        if phase == Phase.val:
            augmentations_intensity, shuffle = 0.0, False

        dataset = AlgalDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            phase=phase.value,
            augmentations_intensity=augmentations_intensity,
            test_size=test_size,
        )
        dataloader[phase] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def define_optimizer(optimizer_name: str, model, lr: float = 4e-3):
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=0.05)
    else:
        raise Exception(
            f"Wrong optimizer name! Expected 'sgd' or 'adam', got {optimizer_name}"
        )

    return optimizer
