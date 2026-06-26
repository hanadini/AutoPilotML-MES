from __future__ import annotations

import pandas as pd

from config.columns import FORCED_PRODUCTION_FEATURES
from config.settings import (
    ENSEMBLE_OVERFIT_PENALTY,
    ENSEMBLE_WEIGHTS,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    TIME_CV_MIN_TRAIN_RATIO,
    TIME_CV_N_FOLDS,
    TIME_CV_VAL_RATIO,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_TUNING_ITERATIONS,
    RANDOM_STATE,
)

from config.settings import N_JOBS
from features.selectors import select_top_correlated_features
from models.ensemble import evaluate_ensemble
from models.registry import get_regression_model
from models.time_cv import TimeFoldResult, run_time_series_cv

from models.xgb_tuner import tune_xgb_regressor
from utils.decorators import pipeline_step


def _select_features_for_fold(
    train_df: pd.DataFrame,
    candidate_feature_columns: list[str],
    target_column: str,
) -> list[str]:
    selected_feature_columns = select_top_correlated_features(
        train_df,
        targets=[target_column],
        top_n=30,
    )

    if "operPressSpeed" in selected_feature_columns:
        selected_feature_columns.remove("operPressSpeed")

    allowed_features = set(candidate_feature_columns)

    for feature in FORCED_PRODUCTION_FEATURES:
        if feature in train_df.columns:
            allowed_features.add(feature)

    selected_feature_columns = [
        col
        for col in selected_feature_columns
        if col in allowed_features
    ]

    if not selected_feature_columns:
        raise ValueError("No features selected after fold feature selection.")

    return selected_feature_columns


def _fit_rf_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    candidate_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    selected_features = _select_features_for_fold(
        train_df=train_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    X_train = train_df[selected_features]
    y_train = train_df[target_column]

    X_val = val_df[selected_features]
    y_val = val_df[target_column]

    model = get_regression_model(
        model_name="rf",
        model_params={
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
    )

    model.fit(X_train, y_train)

    y_pred = pd.Series(
        model.predict(X_val),
        index=y_val.index,
    )

    return y_val, y_pred


def _fit_xgb_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    candidate_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    selected_features = _select_features_for_fold(
        train_df=train_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    X_train = train_df[selected_features]
    y_train = train_df[target_column]

    X_val = val_df[selected_features]
    y_val = val_df[target_column]

    best_params, _ = tune_xgb_regressor(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_iter=XGB_TUNING_ITERATIONS,
        random_state=RANDOM_STATE,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
    )

    model = get_regression_model(
        model_name="xgb",
        model_params=best_params,
    )

    imputer = model.named_steps["imputer"]
    xgb_model = model.named_steps["model"]

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    xgb_model.fit(
        X_train_imp,
        y_train,
        verbose=False,
    )

    y_pred = pd.Series(
        xgb_model.predict(X_val_imp),
        index=y_val.index,
    )

    return y_val, y_pred


def _fit_ensemble_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    candidate_feature_columns: list[str],
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    y_true_rf, y_pred_rf = _fit_rf_fold(
        train_df=train_df,
        val_df=val_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    y_true_xgb, y_pred_xgb = _fit_xgb_fold(
        train_df=train_df,
        val_df=val_df,
        candidate_feature_columns=candidate_feature_columns,
        target_column=target_column,
    )

    if not y_true_rf.index.equals(y_true_xgb.index):
        raise ValueError("RF and XGB fold validation indices do not match.")

    _, best_weight = evaluate_ensemble(
        y_true=y_true_rf,
        pred_a=y_pred_xgb,
        pred_b=y_pred_rf,
        weights=ENSEMBLE_WEIGHTS,
        overfit_penalty=ENSEMBLE_OVERFIT_PENALTY,
    )

    y_pred_ensemble = pd.Series(
        best_weight * y_pred_xgb.values
        + (1.0 - best_weight) * y_pred_rf.values,
        index=y_true_rf.index,
    )

    return y_true_rf, y_pred_ensemble


@pipeline_step("Run model CV summary")
def run_model_cv_summary(
    *,
    df: pd.DataFrame,
    time_column: str,
    candidate_feature_columns: list[str],
    target_column: str,
    model_kind: str,
    n_folds: int = TIME_CV_N_FOLDS,
) -> tuple[list[TimeFoldResult], dict[str, float]]:
    model_kind = model_kind.lower().strip()

    if model_kind == "rf":

        def train_and_predict_fn(
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
        ) -> tuple[pd.Series, pd.Series]:
            return _fit_rf_fold(
                train_df=train_df,
                val_df=val_df,
                candidate_feature_columns=candidate_feature_columns,
                target_column=target_column,
            )

    elif model_kind == "xgb":

        def train_and_predict_fn(
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
        ) -> tuple[pd.Series, pd.Series]:
            return _fit_xgb_fold(
                train_df=train_df,
                val_df=val_df,
                candidate_feature_columns=candidate_feature_columns,
                target_column=target_column,
            )

    elif model_kind == "ensemble_xgb_rf":

        def train_and_predict_fn(
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
        ) -> tuple[pd.Series, pd.Series]:
            return _fit_ensemble_fold(
                train_df=train_df,
                val_df=val_df,
                candidate_feature_columns=candidate_feature_columns,
                target_column=target_column,
            )

    else:
        raise ValueError(
            f"Unsupported model_kind='{model_kind}'. "
            "Supported: ['rf', 'xgb', 'ensemble_xgb_rf']"
        )

    return run_time_series_cv(
        df=df,
        time_column=time_column,
        train_and_predict_fn=train_and_predict_fn,
        n_folds=n_folds,
        min_train_ratio=TIME_CV_MIN_TRAIN_RATIO,
        val_ratio=TIME_CV_VAL_RATIO,
    )