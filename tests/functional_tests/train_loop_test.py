import pytest
from omegaconf import DictConfig, OmegaConf

from src.train.classificator.train import Trainer


class TestTrainLoop:
    @pytest.fixture(scope="class")
    def set_conf(self, data_dir, csv_path) -> DictConfig:
        conf = OmegaConf.create(
            {
                "train": {
                    "log_clearml": False,
                    "epochs": 3,
                    "task_name": "test",
                    "model_save_path": "",
                },
                "dataloader": {
                    "data_dir": data_dir,
                    "csv_path": csv_path,
                    "augmentations_intensity": 0.5,
                    "test_size": 10,
                    "batch_size": 2,
                    "weighted_sampler": True,
                    "save_preprocessed": "",
                },
                "net": {
                    "resume_weights": "",
                    "freeze_grads": False,
                    "pretrained": False,
                    "model_name": "resnet18",
                    "new_in_channels": 6,
                },
                "optimizer": {"optimizer_name": "adam", "lr": 0.0003},
                "scheduler": {"scheduler": False, "t0": 1, "t_mult": 2},
            }
        )
        return conf

    def test_loss_decreasing(self, set_conf):
        trainer = Trainer(set_conf)
        loss1 = trainer.train_one_epoch(0)
        loss2 = 10.0
        for i in range(1, 10):
            loss2 = trainer.train_one_epoch(i)

        assert loss1 > loss2

    def test_deterministic(self, set_conf):
        trainer1 = Trainer(set_conf)
        loss1 = trainer1.train_model()
        trainer2 = Trainer(set_conf)
        loss2 = trainer2.train_model()

        assert loss1 == loss2
