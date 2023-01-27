from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
from loguru import logger
from scipy import interpolate

from src.config import data_config


def add_meta_channels(meta_path: Path, image: np.ndarray, uid: str):
    # image shape: H x W x C
    meta_full_path = meta_path / f"{uid}_metadata.json"
    with open(meta_full_path) as f:
        info = json.load(f)

    if info["s_platform"] is not None:
        prefix = "s"
    elif info["l_platform"] is not None:
        prefix = "l"
    else:
        meta_channels = np.zeros((image.shape[0], image.shape[1], len(data_config.meta_keys)))
        image = np.concatenate((image, meta_channels), axis=2)
        return image

    for key in data_config.meta_keys:
        meta_channels = np.full((image.shape[0], image.shape[1], 1), info[f"{prefix}_{key}"])
        try:
            meta_channels[np.isnan(meta_channels)] = 0.0
            meta_channels[np.isinf(meta_channels)] = 0.0
        except:
            return meta_channels
        image = np.concatenate((image, meta_channels), axis=2)

    return image


def array_inpainting(array: np.ndarray) -> np.ndarray:
    # image shape: H x W x C
    inpainted = deepcopy(array)
    for channel in range(array.shape[-1]):
        img = array[..., channel]
        if np.isnan(img).any() or np.isinf(img).any():
            valid_mask = ~(np.isnan(img) | np.isinf(img))
            coords = np.array(np.nonzero(valid_mask)).T
            values = img[valid_mask]
            try:
                it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
                filled = it(list(np.ndindex(img.shape))).reshape(img.shape)
                inpainted[..., channel] = filled
            except Exception:
                img[np.isnan(img)] = 0.0
                img[np.isinf(img)] = 0.0
                inpainted[..., channel] = img

    return inpainted


def read_dataframe(
    data_dir: Path,
    csv_path: Path | pd.DataFrame,
    phase: str | None = None,
    test_size: int = 0,
) -> (pd.DataFrame, pd.DataFrame):
    images = data_dir.rglob("*.npz")

    images_dict = defaultdict()
    origin = defaultdict()
    for image in images:
        filename_info = str(image.stem).split("_")
        origin[filename_info[0]] = filename_info[1]
        images_dict[filename_info[0]] = image

    df_full = pd.read_csv(csv_path) if isinstance(csv_path, (Path, str)) else csv_path
    df_split = df_full if phase is None else df_full[df_full["split"] == phase]
    data = df_split if test_size <= 0 else df_split.iloc[:test_size]
    try:
        data["filepath"] = data.loc[:, "uid"].map(images_dict)
        data["origin"] = data.loc[:, "uid"].map(origin)
    except KeyError as e:
        logger.warning(f"Not all data were downloaded:\n{e}")
        data["filepath"] = data.loc[:, "uid"].apply(
            lambda x: images_dict[x] if x in images_dict.keys() else None
        )
        data["origin"] = data.loc[:, "uid"].apply(
            lambda x: origin[x] if x in origin.keys() else None
        )

    return data, df_full