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


class Phase(Enum):
    train = "train"
    val = "validation"
    test = "test"


class Origin(Enum):
    landsat = "landsat"
    sentinel = "sentinel"


@dataclass()
class DataConfig:
    meta_keys = ["cloud_cover", "sun_azimuth", "sun_elevation"]
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
    best_features = [
        "nanmean_7days",
        "nanstd_7days",
        "nanmin_7days",
        "nanmax_7days",
        "coef_7days",
        "temp_ge15_7days",
        "temp_ge20_7days",
        "temp_ge25_7days",
        "temp_ge30_7days",
        "temp_le10_7days",
        "perc90_7days",
        "perc10_7days",
        "nanmean_14days",
        "nanstd_14days",
        "nanmin_14days",
        "nanmax_14days",
        "coef_14days",
        "temp_ge15_14days",
        "temp_ge20_14days",
        "temp_ge25_14days",
        "temp_ge30_14days",
        "temp_le5_14days",
        "perc90_14days",
        "perc10_14days",
        "nanmean_30days",
        "nanstd_30days",
        "nanmin_30days",
        "nanmax_30days",
        "coef_30days",
        "temp_ge15_30days",
        "temp_ge20_30days",
        "temp_ge25_30days",
        "temp_ge30_30days",
        "temp_le10_30days",
        "temp_le5_30days",
        "perc90_30days",
        "perc10_30days",
        "nanmean_90days",
        "nanstd_90days",
        "nanmin_90days",
        "nanmax_90days",
        "coef_90days",
        "temp_ge15_90days",
        "temp_ge20_90days",
        "temp_ge25_90days",
        "temp_ge30_90days",
        "temp_le10_90days",
        "temp_le5_90days",
        "temp_le0_90days",
        "temp_lem5_90days",
        "perc90_90days",
        "perc10_90days",
    ]


system_config = SystemConfig()
torch_config = TorchConfig()
data_config = DataConfig()


@dataclass
class NetConfig:
    outputs = 1
    in_channels = 6
    label_column = "density"


net_config = NetConfig()
