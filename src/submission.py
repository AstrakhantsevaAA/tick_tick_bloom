import pandas as pd
import torch
import typer
from einops import asnumpy
from tqdm import tqdm

from src.config import Phase, system_config, torch_config
from src.metrics import weighted_rmse
from src.nets.define_net import define_net
from src.train.classificator.train_utils import create_dataloader


@torch.no_grad()
def prediction(model, dataloader):
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
    csv_path: str = "submission_format.csv",
    model_path: str = "/home/alenaastrakhantseva/PycharmProjects/tick_tick_bloom/models/sgd_0_0001/model_best.pth",
):
    data = pd.read_csv(system_config.data_dir / csv_path)
    if "split" not in data:
        data["split"] = "validation"

    model = define_net("resnet18", weights=model_path)
    dataloader = create_dataloader(
        system_config.data_dir / "benchmark/image_arrays", data, inference=True
    )
    predictions = prediction(model, dataloader[Phase.val])
    predictions.to_csv(
        system_config.data_dir / "benchmark/output/prediction_validation.csv",
        index=False,
    )
    weighted_rmse(predictions)

    submission = predictions.loc[:, ["uid", "region", "pred_int"]]
    submission.columns = ["uid", "region", "severity"]
    submission.to_csv(
        system_config.data_dir / "benchmark/output/submission.csv", index=False
    )


if __name__ == "__main__":
    typer.run(main)
