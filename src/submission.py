import pandas as pd
import typer
from src.config import system_config, torch_config, Phase
from src.nets.define_net import define_net
import torch
from src.train.classificator.train_utils import create_dataloader
from einops import asnumpy
from src.metrics import weighted_rmse


@torch.no_grad()
def prediction(model, dataloader):
    output = {"uid": [], "pred": [], "severity": [], "region": []}

    for batch in dataloader:
        logits = model.forward(batch["image"].to(torch_config.device))
        output["uid"].extend(batch["uid"])
        output["pred"].extend(asnumpy(logits).squeeze())
        output["severity"].extend(asnumpy(batch["severity"]))
        output["region"].extend(batch["region"])

    df_output = pd.DataFrame(output)

    return df_output


def main(
        csv_path: str = "benchmark/uid_train.csv",
        model_path: str = "/home/alenaastrakhantseva/PycharmProjects/tick_tick_bloom/models/resnet_18_adam/model.pth"
):
    data = pd.read_csv(system_config.data_dir / csv_path)
    if "split" not in data:
        data["split"] = "validation"

    model = define_net("resnet18", weights=model_path)
    dataloader = create_dataloader(system_config.data_dir / "benchmark/image_arrays", data)
    predictions = prediction(model, dataloader[Phase.val])
    predictions.to_csv(system_config.data_dir / "benchmark/output/prediction_validation.csv")
    weighted_rmse(predictions)


if __name__ == "__main__":
    typer.run(main)
