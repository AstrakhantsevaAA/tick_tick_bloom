import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import Phase, system_config
from src.data_utils.dataset import AlgalDataset
from src.train.classificator.sampler import define_sampler


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
    weighted_sampler: bool = False,
    save_preprocessed: str | Path | None = None,
    inpaint: bool = False,
    hrrr: bool = False,
    meta_channels_path: Path | None = None,
) -> DefaultDict[Phase, DataLoader]:
    fix_seeds()
    dataloader = defaultdict()

    if csv_path is None:
        raise Exception(
            "csv files with train and validation data are None, for training those files are necessary"
        )

    shuffle, sampler = True, None
    phases = [Phase.test] if inference else [Phase.train, Phase.val]

    for phase in phases:
        if phase in [Phase.val, Phase.test]:
            augmentations_intensity, shuffle, sampler = 0.0, False, None

        dataset = AlgalDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            phase=phase.value,
            augmentations_intensity=augmentations_intensity,
            test_size=test_size,
            inference=inference,
            save_preprocessed=save_preprocessed,
            inpaint=inpaint,
            hrrr=hrrr,
            meta_channels_path=meta_channels_path,
        )

        if phase == Phase.train:
            sampler = define_sampler(dataset) if weighted_sampler else None
            shuffle = True if sampler is None else False

        dataloader[phase] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )

    return dataloader


def define_optimizer(optimizer_name: str, model, lr: float = 4e-3) -> Any:
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "radam":
        optimizer = optim.RAdam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-2)
    else:
        raise Exception(
            f"Wrong optimizer name! Expected 'sgd' or 'adam', got {optimizer_name}"
        )

    return optimizer


def define_scheduler(optimizer: Any, params: dict):
    if params["scheduler_name"] == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params["t0"],
            T_mult=params["t_mult"],
            eta_min=0.000001,
        )
    elif params["scheduler_name"] == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=0.000001)
