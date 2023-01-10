from pathlib import Path

from src.data_utils.dataset import AlgalDataset


def test_datasetclass(data_dir, csv_path):
    dataset = AlgalDataset(
        Path(data_dir),
        Path(csv_path),
        phase="train",
    )
    sample = dataset[14]

    print(sample["image"].shape)
    print(sample["label"])
    print(sample["label_origin"])

    assert len(sample["image"].shape) == 3
    assert sample["image"].shape[0] == 3
    assert round(float(sample["label"]), 4) == 1.7047
