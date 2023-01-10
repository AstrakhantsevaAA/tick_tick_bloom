import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from src.config import system_config


def weighted_rmse(data: pd.DataFrame):
    region_scores = []
    for region in data.region.unique():
        sub = data[data.region == region]
        region_rmse = mean_squared_error(sub.severity, sub.pred_int.values, squared=False)
        print(f"RMSE for {region} (n={len(sub)}): {round(region_rmse, 4)}")
        region_scores.append(region_rmse)

    overall_rmse = np.mean(region_scores)
    print(f"Final score: {overall_rmse}")

    return overall_rmse


if __name__ == "__main__":
    df = pd.read_csv(system_config.data_dir / "benchmark/output/prediction_validation.csv")
    weighted_rmse(df)