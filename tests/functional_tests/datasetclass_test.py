from src.config import data_config, net_config


def test_datasetclass(dataset):
    sample = dataset[199]

    print(sample["image"].shape)
    print(sample["label"])
    print(sample["label_origin"])
    print(sample["hrrr"])
    print(sample)

    assert len(sample["image"].shape) == 3
    assert sample["image"].shape[0] == net_config.in_channels
    assert len(sample["hrrr"]) == len(data_config.best_features)
