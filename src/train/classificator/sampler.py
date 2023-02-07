import pandas as pd
from loguru import logger
from torch.utils.data import WeightedRandomSampler

from src.data_utils.dataset import AlgalDataset


def calculate_weights(df: pd.DataFrame) -> dict:
    weights = dict(df.groupby("region")["uid"].agg("count").astype(float) ** -1)
    w_sum = sum(weights.values())
    weights = {k: 4 * v / w_sum for k, v in weights.items()}
    logger.info(f"Weights for {df.loc[:, 'split'].values[0]}: {weights}")
    return weights


def define_sampler(dataset: AlgalDataset) -> WeightedRandomSampler:
    df_full = dataset.df_full
    df_train = df_full[df_full["split"] == "train"]
    df_val = df_full[df_full["split"] == "validation"]
    df_test = df_full[df_full["split"] == "test"]

    weights_train = calculate_weights(df_train)
    weights_val = calculate_weights(df_val)
    weights_test = calculate_weights(df_test)

    weights_adv = {k: v / weights_test[k] for k, v in weights_train.items()}
    logger.info(f"Adversarial weights: {weights_adv}")

    sampler = WeightedRandomSampler(
        [weights_adv[x] for x in dataset.regions],
        len(dataset.regions),
    )

    return sampler
