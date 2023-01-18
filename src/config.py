import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch


@dataclass
class SystemConfig:
    root_dir: Path = Path(__file__).parents[1]
    model_dir: Path = root_dir / "models"
    data_dir: Path = root_dir / "data"


@dataclass
class TorchConfig:
    if os.getenv("FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


@dataclass
class NetConfig:
    outputs = 1
    in_channels = 6
    label_column = "density"


class Phase(Enum):
    train = "train"
    val = "validation"
    test = "test"


class Origin(Enum):
    landsat = "landsat"
    sentinel = "sentinel"


@dataclass()
class Statistics:
    mean = {
        Origin.landsat: [
            2.20399932e-01,
            2.22531944e-01,
            2.13199551e-01,
            1.97883284e-01,
            1.25201139e-01,
            5.79306179e-01,
        ],
        Origin.sentinel: [
            0.0230814,
            0.02424073,
            0.02255468,
            0.42252148,
            0.16420172,
            0.47051966,
        ],
    }
    std = {
        Origin.landsat: [
            0.03625107,
            0.03315592,
            0.0332012,
            0.07700747,
            0.05879639,
            11.76015284,
        ],
        Origin.sentinel: [
            0.00969575,
            0.0089022,
            0.00891135,
            0.20993949,
            0.14255269,
            8.3464367,
        ],
    }


system_config = SystemConfig()
torch_config = TorchConfig()
net_config = NetConfig()
statistics = Statistics()
