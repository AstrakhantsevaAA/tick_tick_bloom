from src.config import net_config


def test_datasetclass(dataset):
    sample = dataset[14]

    print(sample["image"].shape)
    print(sample["label"])
    print(sample["label_origin"])

    assert len(sample["image"].shape) == 3
    assert sample["image"].shape[0] == net_config.in_channels
