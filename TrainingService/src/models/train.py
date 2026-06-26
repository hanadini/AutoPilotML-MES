from __future__ import annotations

from typing import Any

import pandas as pd

from config.columns import PRIMARY_TARGETs, TIME_COLUMN
from features.selectors import get_available_features
from models.metrics import build_regression_metrics
from utils.decorators import log_dataframe_shape, pipeline_step


@pipeline_step("Prepare modeling dataset")
@log_dataframe_shape("Modeling features")
def prepare_modeling_dataset(
    df: pd.DataFrame,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare X and y for multi-target modeling.
    """
    available_features = get_available_features(df)

    if selected_features is not None:
        available_features = [
            col
            for col in selected_features
            if col in available_features
        ]

    if not available_features:
        raise ValueError("No usable features were selected.")

    modeling_df = (
        df.copy()
        .sort_values(by=TIME_COLUMN)
        .reset_index(drop=True)
    )

    X = modeling_df[available_features].copy()
    y = modeling_df[PRIMARY_TARGETs].copy()

    for target in PRIMARY_TARGETs:
        y[target] = pd.to_numeric(
            y[target],
            errors="coerce",
        )

    valid_idx = y.notna().all(axis=1)

    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X, y, available_features


@pipeline_step("Evaluate multi-target regression model")
def evaluate_regression_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Evaluate multi-target regression model.
    """
    predictions = model.predict(X_test)

    results: dict[str, dict[str, float]] = {}

    for index, target in enumerate(y_test.columns):
        y_true = y_test.iloc[:, index]
        y_pred = predictions[:, index]

        results[target] = build_regression_metrics(
            y_true,
            y_pred,
        )

    return results


@pipeline_step("Extract feature importance")
def extract_feature_importance(
    model_object: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract feature importance from models that expose
    feature_importances_.

    Supports:
    - Pipeline(imputer + model)
    - raw model
    - RandomForestRegressor
    - XGBRegressor
    - LGBMRegressor
    """
    if hasattr(model_object, "named_steps"):
        final_model = model_object.named_steps["model"]
    else:
        final_model = model_object

    if not hasattr(final_model, "feature_importances_"):
        raise ValueError(
            f"Model '{type(final_model).__name__}' "
            "does not expose feature_importances_."
        )

    importances = final_model.feature_importances_

    if len(importances) != len(feature_names):
        min_len = min(
            len(importances),
            len(feature_names),
        )

        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    return (
        importance_df
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )