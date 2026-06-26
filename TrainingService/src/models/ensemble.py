from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import ENSEMBLE_OVERFIT_PENALTY
from models.metrics import build_regression_metrics
from utils.decorators import pipeline_step


@pipeline_step("Evaluate ensemble")
def evaluate_ensemble(
    *,
    y_true: pd.Series,
    pred_a,
    pred_b,
    weights: list[float],
    overfit_penalty: ENSEMBLE_OVERFIT_PENALTY,
) -> tuple[dict[str, float], float]:
    """
    Evaluate weighted ensemble on validation data.

    blended_prediction =
        weight * pred_a
        + (1 - weight) * pred_b

    Slightly penalizes extreme weights
    to reduce over-trusting one model.
    """
    best_metrics: dict[str, float] | None = None
    best_weight: float | None = None

    best_score = -np.inf

    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    for weight in weights:
        blended_pred = (
            weight * pred_a
            + (1.0 - weight) * pred_b
        )

        metrics = build_regression_metrics(
            y_true,
            blended_pred,
        )

        weight_penalty = (
            overfit_penalty
            * abs(weight - 0.5)
        )

        score = metrics["r2"] - weight_penalty

        if score > best_score:
            best_score = score
            best_weight = weight
            best_metrics = metrics

    if best_metrics is None or best_weight is None:
        raise ValueError(
            "No valid ensemble weight was evaluated."
        )

    return best_metrics, best_weight