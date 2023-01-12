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
def prediction(model: Any, dataloader: DataLoader) -> pd.DataFrame:
    output = {"uid": [], "pred_raw": [], "pred_int": [], "severity": [], "region": []}

    for batch in tqdm(dataloader):
        logits = model.forward(batch["image"].to(torch_config.device))
        output["uid"].extend(batch["uid"])
        output["pred_raw"].extend(asnumpy(logits).squeeze())
        output["pred_int"].extend((asnumpy(logits).squeeze()).astype(int))
        output["severity"].extend(asnumpy(batch["severity"]))
        output["region"].extend(batch["region"])

    df_output = pd.DataFrame(output)

    return df_output


def main(
    csv_path: str = "splits/balanced_validation/dumb_split_full_df.csv",
    model_path: str = "weighted_sampler_300epoch/model_best.pth",
    inference: bool = False,
):
    outputs_save_path = (
        system_config.data_dir
        / f"outputs/{model_path.split('/')[-2]}_{csv_path.split('/')[-1]}"
    )
    outputs_save_path.mkdir(parents=True, exist_ok=True)

    model = define_net("resnet18", weights=system_config.model_dir / model_path)
    dataloader = create_dataloader(
        system_config.data_dir / "benchmark/image_arrays",
        system_config.data_dir / csv_path,
        inference=inference,
    )
    predictions = prediction(model, dataloader[Phase.val])
    predictions.to_csv(
        outputs_save_path / "prediction_validation.csv",
        index=False,
    )

    weighted_rmse(predictions)

    if inference:
        submission = predictions.loc[:, ["uid", "region", "pred_int"]]
        submission.columns = ["uid", "region", "severity"]
        submission.to_csv(
            outputs_save_path / "submission.csv",
            index=False,
        )


if __name__ == "__main__":
    typer.run(main)
