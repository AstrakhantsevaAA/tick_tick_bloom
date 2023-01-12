from pathlib import Path
from typing import Any, Optional

import hydra
import torch
from clearml import Task
from loguru import logger
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.config import Phase, net_config, system_config, torch_config
from src.metrics import weighted_rmse
from src.nets.define_net import define_net
from src.submission import prediction
from src.train.classificator.loss import DensityMSELoss
from src.train.classificator.train_utils import (
    create_dataloader,
    define_optimizer,
    fix_seeds,
)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.task = (
            Task.init(
                project_name="tick_tick_bloom/train", task_name=cfg.train.task_name
            )
            if cfg.train.log_clearml
            else None
        )
        self.clearml_logger = None if self.task is None else self.task.get_logger()
        self.epochs = cfg.train.epochs
        self.model_save_path = (
            system_config.model_dir / cfg.train.model_save_path
            if len(cfg.train.model_save_path) > 0
            else None
        )

        fix_seeds()
        self.dataloader = create_dataloader(
            data_dir=Path(cfg.dataloader.data_dir),
            csv_path=Path(cfg.dataloader.csv_path),
            augmentations_intensity=cfg.dataloader.augmentations_intensity,
            batch_size=cfg.dataloader.batch_size,
            test_size=cfg.dataloader.test_size,
            weighted_sampler=cfg.dataloader.weighted_sampler,
        )
        self.train_iters = len(self.dataloader[Phase.train])
        self.val_iters = len(self.dataloader[Phase.val])

        self.model = define_net(
            model_name=cfg.net.model_name,
            freeze_grads=cfg.net.freeze_grads,
            outputs=net_config.outputs,
            pretrained=cfg.net.pretrained,
            weights=cfg.net.resume_weights,
        )

        self.criterion = DensityMSELoss()
        self.optimizer = define_optimizer(
            cfg.optimizer.optimizer_name, self.model, cfg.optimizer.lr
        )
        if cfg.scheduler.scheduler:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=cfg.scheduler.t0,
                T_mult=cfg.scheduler.t_mult,
                eta_min=0.000001,
            )
        else:
            self.scheduler = None

    @torch.no_grad()
    def evaluation(
        self,
        epoch: int = -1,
        phase: Any = "val",
    ) -> Optional:
        self.model.eval()
        logger.info(f"Starting {phase} epoch {epoch}")
        predictions, running_loss = prediction(
            self.model, self.dataloader[Phase.val], self.criterion
        )
        loss = running_loss / self.val_iters
        overall_rmse, region_scores = weighted_rmse(predictions)

        if self.clearml_logger is not None:
            self.clearml_logger.report_scalar(
                f"Loss", phase, iteration=epoch, value=loss
            )
            self.clearml_logger.report_scalar(
                f"Overall rMSE", phase, iteration=epoch, value=overall_rmse
            )
            for k, v in region_scores.items():
                self.clearml_logger.report_scalar(
                    f"rMSE {k}", phase, iteration=epoch, value=v
                )

        return loss, overall_rmse

    def train_one_epoch(
        self,
        epoch: int,
    ):
        self.model.train()
        running_loss = 0

        logger.info(f"Starting training epoch {epoch}")
        for batch_n, batch in tqdm(
            enumerate(self.dataloader[Phase.train]), total=self.train_iters
        ):
            self.optimizer.zero_grad()
            outputs = self.model(batch["image"].to(torch_config.device))
            loss = self.criterion(outputs, batch["label"].to(torch_config.device))
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if self.clearml_logger is not None:
                self.clearml_logger.report_scalar(
                    f"Running_loss",
                    "train",
                    iteration=(epoch + 1) * batch_n,
                    value=running_loss / (batch_n + 1),
                )
        loss_total = running_loss / self.train_iters
        if self.clearml_logger is not None:
            self.clearml_logger.report_scalar(
                "Loss", "train", iteration=epoch, value=loss_total
            )
            self.clearml_logger.report_scalar(
                "LR",
                "train",
                iteration=epoch,
                value=self.optimizer.param_groups[0]["lr"],
            )

        return loss_total

    def train_model(self):
        fix_seeds()
        loss = 0.0
        best_rmse = 10.0

        for epoch in range(self.epochs):
            loss = self.train_one_epoch(epoch)
            val_loss, overall_rmse = self.evaluation(
                epoch,
                phase=Phase.val.value,
            )

            if self.scheduler:
                self.scheduler.step()

            if self.model_save_path:
                model_save_path = self.model_save_path
                model_save_path.mkdir(exist_ok=True, parents=True)
                torch.save(
                    self.model,
                    self.model_save_path / "model.pth",
                )
                if overall_rmse < best_rmse:
                    best_rmse = overall_rmse
                    torch.save(
                        self.model,
                        self.model_save_path / "model_best.pth",
                    )
                    logger.success(
                        f"Saving best model to {model_save_path} as model_best.pth"
                    )

        if self.clearml_logger is not None:
            self.clearml_logger.flush()

        return loss


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.train_model()


if __name__ == "__main__":
    main()
