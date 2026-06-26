from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
)

from utils.decorators import pipeline_step


@pipeline_step("Build regression metrics")
def build_regression_metrics(
    y_true: pd.Series,
    y_pred,
) -> dict[str, float]:

    mse = mean_squared_error(y_true, y_pred)

    return {
        "loss_mse": float(mse),

        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),

        "median_ae": float(
            median_absolute_error(y_true, y_pred)
        ),

        "rmse": float(
            np.sqrt(mse)
        ),

        "mape_pct": float(
            mean_absolute_percentage_error(
                y_true,
                y_pred
            ) * 100
        ),

        "r2": float(
            r2_score(y_true, y_pred)
        ),
    }