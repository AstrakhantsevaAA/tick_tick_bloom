import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_squared_error

from src.config import system_config


def weighted_rmse(data: pd.DataFrame) -> (float, dict):
    regions = data.region.unique()
    region_scores = {k: [] for k in regions}
    for region in regions:
        sub = data[data.region == region]
        region_rmse = mean_squared_error(
            sub.severity, sub.pred_int.values, squared=False
        )
        logger.info(f"RMSE for {region} (n={len(sub)}): {round(region_rmse, 4)}")
        region_scores[region] = region_rmse

    overall_rmse = np.mean(list(region_scores.values()))
    logger.info(f"Final score: {overall_rmse}")
    return overall_rmse, region_scores


if __name__ == "__main__":
    df = pd.read_csv(
        system_config.data_dir / "benchmark/output/prediction_validation.csv"
    )
    weighted_rmse(df)
