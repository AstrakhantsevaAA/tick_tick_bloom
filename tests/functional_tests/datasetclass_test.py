def test_datasetclass(dataset):
    sample = dataset[14]

    print(sample["image"].shape)
    print(sample["label"])
    print(sample["label_origin"])

    assert len(sample["image"].shape) == 3
    assert sample["image"].shape[0] == 3
    assert round(float(sample["label"]), 4) == 3.3988
