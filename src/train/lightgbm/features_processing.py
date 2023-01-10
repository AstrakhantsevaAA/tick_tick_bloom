import pandas as pd
import typer
from src.config import system_config, torch_config, Phase
from src.nets.define_net import define_net
import torch
from src.train.classificator.train_utils import create_dataloader
from einops import asnumpy
from src.metrics import weighted_rmse


@torch.no_grad()
def main(csv_path: str = "benchmark/uid_train.csv", model_path: str = "/home/alenaastrakhantseva/PycharmProjects/tick_tick_bloom/models/resnet_18_adam/model.pth"):
    model = define_net("resnet18", weights=model_path)

    dataloader = create_dataloader(system_config.data_dir / "benchmark/image_arrays", system_config.data_dir / csv_path)

    output = {"uid": [], "pred": [], "severity": [], "region": []}

    for batch in dataloader[Phase.val]:
        logits = model.forward(batch["image"].to(torch_config.device))
        output["uid"].extend(batch["uid"])
        output["pred"].extend(asnumpy(logits).squeeze())
        output["severity"].extend(asnumpy(batch["severity"]))
        output["region"].extend(batch["region"])

    df_output = pd.DataFrame(output)
    df_output["pred"] = df_output["pred"].values.reshape(-1)
    df_output.to_csv(system_config.data_dir / "benchmark/output/prediction_validation.csv")
    weighted_rmse(df_output)


if __name__ == "__main__":
    typer.run(main)
