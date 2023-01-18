from typing import Any

import pandas as pd
import typer
from einops import asnumpy
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Phase, system_config, torch_config
from src.metrics import weighted_rmse
from src.nets.define_net import define_net
from src.train.classificator.train_utils import create_dataloader


@no_grad()
def prediction(
    model: Any, dataloader: DataLoader, criterion: Any | None = None
) -> (pd.DataFrame, float):
    model.eval()
    running_loss = 0.0
    output = {"uid": [], "pred_raw": [], "pred_int": [], "severity": [], "region": []}

    for batch in tqdm(dataloader):
        logits = model.forward(batch["image"].to(torch_config.device))
        output["uid"].extend(batch["uid"])
        output["pred_raw"].extend(asnumpy(logits).squeeze())
        output["pred_int"].extend((asnumpy(logits).squeeze()).astype(int))
        output["severity"].extend(asnumpy(batch["severity"]))
        output["region"].extend(batch["region"])

        if criterion:
            loss = criterion(logits, batch["label"].to(torch_config.device))
            running_loss += loss.item()

    df_output = pd.DataFrame(output)

    return df_output, running_loss


def main(
    csv_path: str = "splits/downloaded.csv",
    model_path: str = "new_data_6_channels_norm/model_best.pth",
    inference: bool = True,
):
    outputs_save_path = (
        system_config.data_dir
        / f"outputs/{model_path.split('/')[-2]}_{csv_path.split('/')[-1]}"
    )
    outputs_save_path.mkdir(parents=True, exist_ok=True)

    model = define_net("resnet18", weights=system_config.model_dir / model_path)
    dataloader = create_dataloader(
        system_config.data_dir / "arrays/more_arrays_fixed",
        system_config.data_dir / csv_path,
        inference=inference,
    )
    phase = Phase.test if inference else Phase.val
    predictions, _ = prediction(model, dataloader[phase])
    predictions.to_csv(
        outputs_save_path / "prediction_validation.csv",
        index=False,
    )

    if inference:
        submission = predictions.loc[:, ["uid", "region", "pred_int"]]
        submission.columns = ["uid", "region", "severity"]
        submission.to_csv(
            outputs_save_path / "submission.csv",
            index=False,
        )
    else:
        weighted_rmse(predictions)


if __name__ == "__main__":
    typer.run(main)
