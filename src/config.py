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
    label_column = "density"


class Phase(Enum):
    train = "train"
    val = "validation"


system_config = SystemConfig()
torch_config = TorchConfig()
net_config = NetConfig()
