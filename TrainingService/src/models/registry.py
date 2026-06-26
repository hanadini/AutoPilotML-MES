from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from config.settings import (
    BEST_MODELS_REGISTRY_PATH,
    DEFAULT_IMPUTER_STRATEGY,
)
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def get_regression_model(
    model_name: str,
    model_params: dict | None = None,
) -> Any:
    params = model_params or {}
    model_name = model_name.lower().strip()

    if model_name in {"linear_regression", "lr"}:
        model = LinearRegression(**params)

    elif model_name == "ridge":
        model = Ridge(**params)

    elif model_name == "lasso":
        model = Lasso(**params)

    elif model_name in {"random_forest", "rf"}:
        model = RandomForestRegressor(**params)

    elif model_name in {"xgb", "xgboost"}:
        model = XGBRegressor(**params)

    elif model_name in {"lgbm", "lightgbm"}:
        model = LGBMRegressor(**params)

    else:
        supported = [
            "linear_regression",
            "lr",
            "ridge",
            "lasso",
            "random_forest",
            "rf",
            "xgb",
            "xgboost",
            "lgbm",
            "lightgbm",
        ]

        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            f"Supported: {supported}"
        )

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy=DEFAULT_IMPUTER_STRATEGY),
            ),
            ("model", model),
        ]
    )


@pipeline_step("Save best models registry")
def save_best_models_registry(
    all_results,
    output_path: str | Path = BEST_MODELS_REGISTRY_PATH,
) -> dict[str, dict[str, Any]]:
    grouped_results = defaultdict(list)

    for result in all_results:
        grouped_results[result.target_name].append(result)

    best_models: dict[str, dict[str, Any]] = {}

    for target_name, results in grouped_results.items():
        best_result = max(
            results,
            key=lambda r: r.metrics["test"]["r2"],
        )

        registry_entry: dict[str, Any] = {
            "target_name": best_result.target_name,
            "algorithm_name": best_result.algorithm_name,
            "artifact_model_name": best_result.artifact_model_name,
            "artifact_dir": str(best_result.artifact_dir),
            "test_r2": best_result.metrics["test"]["r2"],
            "test_rmse": best_result.metrics["test"]["rmse"],
            "test_mae": best_result.metrics["test"]["mae"],
            "validation_r2": best_result.metrics["validation"]["r2"],
        }

        if best_result.algorithm_name == "ensemble_xgb_rf":
            ensemble_info = best_result.metrics.get("ensemble", {})

            registry_entry["type"] = "weighted_ensemble"
            registry_entry["members"] = {
                "xgb": ensemble_info["members"][0],
                "rf": ensemble_info["members"][1],
            }
            registry_entry["weights"] = {
                "xgb": ensemble_info["xgb_weight"],
                "rf": ensemble_info["rf_weight"],
            }
        else:
            registry_entry["type"] = "single_model"

        best_models[target_name] = registry_entry

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(
            best_models,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Best models registry saved: %s",
        output_file,
    )

    return best_models


def log_best_models_summary(
    best_models: dict[str, dict[str, Any]],
) -> None:
    logger.info("=" * 50)
    logger.info("BEST MODEL PER TARGET")
    logger.info("=" * 50)

    for target, info in best_models.items():
        logger.info(
            "Target=%s | Model=%s | Type=%s | Test R2=%.4f",
            target,
            info["algorithm_name"],
            info["type"],
            info["test_r2"],
        )

        if info["type"] == "weighted_ensemble":
            logger.info(
                "Target=%s | Members=%s | Weights=%s",
                target,
                info["members"],
                info["weights"],
            )