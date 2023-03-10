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


def add_not_loaded(out: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    not_loaded = out.loc[~out["uid"].isin(submission.uid)]
    print(f"Not loaded: {len(not_loaded)}")
    full = pd.concat([submission, not_loaded])
    full = full.sort_values(by=["uid"], ignore_index=True)
    return full


@no_grad()
def prediction(
    model: Any, dataloader: DataLoader, criterion: Any | None = None
) -> (pd.DataFrame, float):
    model.eval()
    running_loss = 0.0
    output = {"uid": [], "pred_raw": [], "pred_int": [], "severity": [], "region": []}

    for batch in tqdm(dataloader):
        if len(batch.get("hrrr")) > 0:
            meta = (
                batch["meta"].to(torch_config.device)
                if len(batch.get("meta")) > 0
                else None
            )
            logits = model(
                batch["image"].to(torch_config.device),
                batch["hrrr"].to(torch_config.device),
                meta,
            )
        else:
            logits = model(batch["image"].to(torch_config.device))
        output["uid"].extend(batch["uid"])
        output["pred_raw"].extend(asnumpy(logits).squeeze())
        output["pred_int"].extend((asnumpy(logits).squeeze()).clip(1, 5).astype(int))
        output["severity"].extend(asnumpy(batch["severity"]))
        output["region"].extend(batch["region"])

        if criterion:
            loss = criterion(logits, batch["label"].to(torch_config.device))
            running_loss += loss.item()

    df_output = pd.DataFrame(output)

    return df_output, running_loss


def main(
    csv_path: str = "splits/hrrr_features_forcasted_scaled.csv",
    model_path: str = "scl_channels/model_best.pth",
    inference: bool = True,
):
    outputs_save_path = (
        system_config.data_dir
        / f"outputs/{model_path.split('/')[-2]}_{csv_path.split('/')[-1]}"
    )
    outputs_save_path.mkdir(parents=True, exist_ok=True)

    model = define_net(
        "rexnet_100",
        weights_resume=system_config.model_dir / model_path,
        hrrr=True,
        new_in_channels=8,
    )
    dataloader = create_dataloader(
        system_config.data_dir / "arrays/more_arrays_fixed",
        system_config.data_dir / csv_path,
        inference=inference,
        save_preprocessed=None,
        inpaint=True,
        hrrr=True,
        meta_channels_path=None,
    )
    phase = Phase.test if inference else Phase.val
    predictions, _ = prediction(model, dataloader[phase])

    out = system_config.data_dir / "outputs"

    if inference:
        submission = predictions.loc[:, ["uid", "region", "pred_int"]]
        submission.columns = ["uid", "region", "severity"]
        submission.to_csv(
            outputs_save_path / "submission.csv",
            index=False,
        )

        out = pd.read_csv(out / "united_lightgbm1_and_best_net.csv")
        full = add_not_loaded(out, submission)
        print(f"Submission saved to: {outputs_save_path}")
        full.to_csv(outputs_save_path / "submission_with_not_loaded.csv", index=False)

    else:
        predictions.to_csv(
            outputs_save_path / "prediction_validation.csv",
            index=False,
        )
        out = pd.read_csv(out / "prediction_validation.csv")
        full = add_not_loaded(out, predictions)
        weighted_rmse(full)


if __name__ == "__main__":
    typer.run(main)
