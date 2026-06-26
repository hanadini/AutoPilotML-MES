from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from config.settings import (
    DEFAULT_IMPUTER_STRATEGY,
    RANDOM_STATE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_PARAM_DISTRIBUTION,
    XGB_TUNING_ITERATIONS,
)
from models.metrics import build_regression_metrics
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@pipeline_step("Tune XGBoost regressor")
def tune_xgb_regressor(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_iter: int = XGB_TUNING_ITERATIONS,
    random_state: int = RANDOM_STATE,
    early_stopping_rounds: int = XGB_EARLY_STOPPING_ROUNDS,
) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Tune XGB on train -> validation only.
    Test set must remain untouched.
    """
    sampled_params = list(
        ParameterSampler(
            param_distributions=XGB_PARAM_DISTRIBUTION,
            n_iter=n_iter,
            random_state=random_state,
        )
    )

    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = -np.inf

    imputer = SimpleImputer(
        strategy=DEFAULT_IMPUTER_STRATEGY,
    )

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    for index, params in enumerate(
        sampled_params,
        start=1,
    ):
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )

        model.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val_imp)

        metrics = build_regression_metrics(
            y_val,
            y_val_pred,
        )

        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        logger.info(
            (
                "XGB tuning %s/%s | "
                "R2=%.4f | RMSE=%.4f | "
                "best_iteration=%s | params=%s"
            ),
            index,
            len(sampled_params),
            metrics["r2"],
            metrics["rmse"],
            best_iteration,
            params,
        )

        if metrics["r2"] > best_score:
            best_score = metrics["r2"]
            best_params = dict(params)
            best_metrics = metrics

            if best_iteration is not None:
                best_params["n_estimators"] = int(best_iteration) + 1

    if best_params is None or best_metrics is None:
        raise RuntimeError(
            "XGB tuning failed to produce best parameters."
        )

    return best_params, best_metrics