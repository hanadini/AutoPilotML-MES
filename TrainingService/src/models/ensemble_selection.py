from __future__ import annotations

from copy import deepcopy
from typing import Any

from config.settings import (
    ENSEMBLE_MIN_GAIN,
    ENSEMBLE_OVERFIT_PENALTY,
    ENSEMBLE_WEIGHTS,
)
from models.ensemble import evaluate_ensemble
from models.metrics import build_regression_metrics
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@pipeline_step("Try build XGB/RF ensemble")
def try_build_xgb_rf_ensemble(
    target: str,
    rf_result: Any,
    xgb_result: Any,
    *,
    weights: list[float] | None = None,
    overfit_penalty: float = ENSEMBLE_OVERFIT_PENALTY,
    min_gain: float = ENSEMBLE_MIN_GAIN,
) -> tuple[Any | None, dict[str, float], dict[str, float], float]:
    weights = weights or ENSEMBLE_WEIGHTS

    val_metrics, best_weight = evaluate_ensemble(
        y_true=rf_result.y_val_true,
        pred_a=xgb_result.y_val_pred,
        pred_b=rf_result.y_val_pred,
        weights=weights,
        overfit_penalty=overfit_penalty,
    )

    best_single_val_r2 = max(
        rf_result.metrics["validation"]["r2"],
        xgb_result.metrics["validation"]["r2"],
    )

    accepted = val_metrics["r2"] > best_single_val_r2 + min_gain

    test_pred = (
        best_weight * xgb_result.y_test_pred
        + (1.0 - best_weight) * rf_result.y_test_pred
    )

    test_metrics = build_regression_metrics(
        rf_result.y_test_true,
        test_pred,
    )

    if not accepted:
        logger.info(
            "Ensemble rejected | target=%s | ensemble_val_r2=%.4f | "
            "best_single_val_r2=%.4f | min_gain=%.4f",
            target,
            val_metrics["r2"],
            best_single_val_r2,
            min_gain,
        )

        return None, val_metrics, test_metrics, best_weight

    logger.info(
        "Ensemble accepted | target=%s | ensemble_val_r2=%.4f | "
        "best_single_val_r2=%.4f | best_weight=%.2f",
        target,
        val_metrics["r2"],
        best_single_val_r2,
        best_weight,
    )

    ensemble_result = deepcopy(xgb_result)

    ensemble_result.algorithm_name = "ensemble_xgb_rf"
    ensemble_result.artifact_model_name = f"{target}_ensemble_xgb_rf_v1"

    ensemble_result.metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "row_counts": xgb_result.metrics["row_counts"],
        "split_type": xgb_result.metrics["split_type"],
        "selected_feature_count": xgb_result.metrics["selected_feature_count"],
        "selected_features": xgb_result.metrics["selected_features"],
        "ensemble": {
            "xgb_weight": best_weight,
            "rf_weight": 1.0 - best_weight,
            "members": [
                xgb_result.artifact_model_name,
                rf_result.artifact_model_name,
            ],
            "selection_basis": "validation_r2",
            "min_gain": min_gain,
        },
    }

    return ensemble_result, val_metrics, test_metrics, best_weight