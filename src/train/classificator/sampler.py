from pathlib import Path

from torch.utils.data import WeightedRandomSampler

from src.data_utils.dataset import AlgalDataset


def calculate_weights(df):
    weights = dict(df.groupby("region")["uid"].agg("count").astype(float) ** -1)
    w_sum = sum(weights.values())
    weights = {k: 4 * v / w_sum for k, v in weights.items()}

    return weights


def define_sampler(dataset):
    df_full = dataset.df_full
    df_train = df_full[df_full["split"] == "train"]
    df_val = df_full[df_full["split"] == "validation"]

    weights_train = calculate_weights(df_train)
    weights_val = calculate_weights(df_val)
    weights = {k: v / weights_val[k] for k, v in weights_train.items()}

    print(weights_train, weights_val, weights)

    sampler = WeightedRandomSampler(
        [weights[x] for x in dataset.regions],
        len(dataset.regions),
    )

    return sampler


if __name__ == "__main__":
    dataset = AlgalDataset(
        Path(
            "/home/alenaastrakhantseva/PycharmProjects/tick_tick_bloom/data/benchmark/image_arrays"
        ),
        Path(
            "/home/alenaastrakhantseva/PycharmProjects/tick_tick_bloom/data/splits/balanced_validation/dumb_split_full_df.csv"
        ),
        phase="train",
    )
    define_sampler(dataset)
