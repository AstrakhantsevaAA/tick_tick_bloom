import os
import torch
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SystemConfig:
    root_dir: Path = Path(__file__).parent.parent
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
    outputs = 5


system_config = SystemConfig()
torch_config = TorchConfig()
net_config = NetConfig()
