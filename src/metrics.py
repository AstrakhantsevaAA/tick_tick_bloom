import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


def weighted_rmse(data: pd.DataFrame):
    region_scores = []
    for region in data.region.unique():
        sub = data[data.region == region]
        region_rmse = mean_squared_error(sub.severity, sub.pred.values.astype(int), squared=False)
        print(f"RMSE for {region} (n={len(sub)}): {round(region_rmse, 4)}")
        region_scores.append(region_rmse)

    overall_rmse = np.mean(region_scores)
    print(f"Final score: {overall_rmse}")

    return overall_rmse
