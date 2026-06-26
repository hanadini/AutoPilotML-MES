from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.settings import (
    LGBM_COLSAMPLE_BYTREE,
    LGBM_LEARNING_RATE,
    LGBM_N_ESTIMATORS,
    LGBM_NUM_LEAVES,
    LGBM_SUBSAMPLE,
    RANDOM_STATE,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS, N_JOBS, VERBOSE,
)


@dataclass(frozen=True)
class AlgorithmConfig:
    algorithm_name: str
    artifact_suffix: str
    model_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def get_algorithm_configs() -> list[AlgorithmConfig]:
    return [
        AlgorithmConfig(
            algorithm_name="rf",
            artifact_suffix="rf_v2",
            model_params={
                "n_estimators": RF_N_ESTIMATORS,
                "max_depth": RF_MAX_DEPTH,
                "random_state": RANDOM_STATE,
                "n_jobs": N_JOBS,
            },
            notes=(
                "MDF1 Random Forest model "
                "with engineered features"
            ),
        ),
        AlgorithmConfig(
            algorithm_name="xgb",
            artifact_suffix="xgb_tuned_v2",
            model_params={},
            notes=(
                "MDF1 tuned XGBoost model "
                "with early stopping"
            ),
        ),
        AlgorithmConfig(
            algorithm_name="lgbm",
            artifact_suffix="lgbm_v1",
            model_params={
                "n_estimators": LGBM_N_ESTIMATORS,
                "learning_rate": LGBM_LEARNING_RATE,
                "num_leaves": LGBM_NUM_LEAVES,
                "subsample": LGBM_SUBSAMPLE,
                "colsample_bytree": LGBM_COLSAMPLE_BYTREE,
                "random_state": RANDOM_STATE,
                "n_jobs": N_JOBS,
                "verbose": VERBOSE,
            },
            notes=(
                "MDF1 LightGBM model "
                "with engineered features"
            ),
        ),
    ]