from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.columns import FORCED_PRODUCTION_FEATURES
from config.settings import (
    IID_TEST_SIZE,
    IID_VALID_SIZE,
    RANDOM_STATE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_TUNING_ITERATIONS,
)
from data.splitter import (
    build_split_date_range_report,
    split_time_based,
    split_train_val_test,
)
from features.selectors import select_top_correlated_features
from models.artifact_saver import save_full_artifact_bundle
from models.metrics import build_regression_metrics
from models.registry import get_regression_model
from models.train import extract_feature_importance
from models.xgb_tuner import tune_xgb_regressor
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingPipelineResult:
    algorithm_name: str
    artifact_model_name: str
    target_name: str
    metrics: dict
    artifact_dir: str
    y_val_true: pd.Series | None
    y_val_pred: pd.Series | None
    y_test_true: pd.Series
    y_test_pred: pd.Series


def _build_effective_candidate_features(
    df: pd.DataFrame,
    candidate_feature_columns: list[str],
) -> list[str]:
    effective_features = list(candidate_feature_columns)

    for feature in FORCED_PRODUCTION_FEATURES:
        if feature in df.columns and feature not in effective_features:
            effective_features.append(feature)

    return effective_features


def _build_split_metadata(
    split_result,
    use_time_based_split: bool,
) -> dict[str, Any]:
    return {
        "row_counts": {
            "train": len(split_result.train_df),
            "val": len(split_result.val_df),
            "test": len(split_result.test_df),
        },
        "split_type": (
            "time_based"
            if use_time_based_split
            else "random"
        ),
    }


def _select_train_only_features(
    train_df: pd.DataFrame,
    target_column: str,
    allowed_features: set[str],
) -> list[str]:
    selected_feature_columns = select_top_correlated_features(
        train_df,
        targets=[target_column],
        top_n=30,
    )

    if "operPressSpeed" in selected_feature_columns:
        selected_feature_columns.remove("operPressSpeed")

    selected_feature_columns = [
        col
        for col in selected_feature_columns
        if col in allowed_features
    ]

    selected_feature_columns = list(
        dict.fromkeys(selected_feature_columns)
    )

    if not selected_feature_columns:
        raise ValueError(
            "No features selected after train-only feature selection."
        )

    return selected_feature_columns


def _build_metrics(
    *,
    split_result,
    use_time_based_split: bool,
    selected_feature_columns: list[str],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    time_column: str | None = None,
    tuning_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        **_build_split_metadata(
            split_result,
            use_time_based_split,
        ),
        "selected_feature_count": len(selected_feature_columns),
        "selected_features": selected_feature_columns,
    }

    if time_column is not None:
        metrics["data_period"] = build_split_date_range_report(
            split_result=split_result,
            time_column=time_column,
        )

    if tuning_info:
        metrics["tuning"] = tuning_info

    return metrics


def _extract_feature_importance_if_supported(
    *,
    model,
    algorithm_name: str,
    selected_feature_columns: list[str],
) -> pd.DataFrame | None:
    if algorithm_name.lower() not in {
        "rf",
        "random_forest",
        "xgb",
        "xgboost",
        "lgbm",
        "lightgbm",
    }:
        return None

    try:
        return extract_feature_importance(
            model,
            selected_feature_columns,
        )
    except Exception as exc:
        logger.warning(
            "Feature importance extraction skipped: %s",
            exc,
        )
        return None


@pipeline_step("Run single target regression pipeline")
def run_single_target_regression_pipeline(
    *,
    df: pd.DataFrame,
    candidate_feature_columns: list[str],
    target_column: str,
    algorithm_name: str,
    artifact_model_name: str,
    model_params: dict | None,
    artifacts_root: str,
    preprocessor: Any | None = None,
    notes: str | None = None,
    random_state: int = RANDOM_STATE,
    use_time_based_split: bool = True,
    time_column: str | None = None,
    save_artifacts: bool = True,
) -> TrainingPipelineResult:
    logger.info(
        "Starting regression pipeline | target=%s | algorithm=%s | artifact=%s | save_artifacts=%s",
        target_column,
        algorithm_name,
        artifact_model_name,
        save_artifacts,
    )

    effective_candidate_features = _build_effective_candidate_features(
        df=df,
        candidate_feature_columns=candidate_feature_columns,
    )

    required_columns = set(
        effective_candidate_features + [target_column]
    )

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in dataframe: {sorted(missing)}"
        )

    selected_columns = effective_candidate_features + [target_column]

    if time_column and time_column in df.columns:
        selected_columns.append(time_column)

    selected_columns = list(dict.fromkeys(selected_columns))

    work_df = (
        df[selected_columns]
        .dropna(subset=[target_column])
        .copy()
    )

    if use_time_based_split:
        if not time_column:
            raise ValueError(
                "time_column must be provided when use_time_based_split=True"
            )

        split_result = split_time_based(
            df=work_df,
            time_column=time_column,
            test_size=IID_TEST_SIZE,
            val_size=IID_VALID_SIZE,
        )
    else:
        split_result = split_train_val_test(
            df=work_df,
            test_size=IID_TEST_SIZE,
            val_size=IID_VALID_SIZE,
            random_state=random_state,
            shuffle=True,
            time_column=time_column,
        )

    train_df = split_result.train_df.copy()
    val_df = split_result.val_df.copy()
    test_df = split_result.test_df.copy()

    allowed_features = set(effective_candidate_features)

    for feature in FORCED_PRODUCTION_FEATURES:
        if feature in train_df.columns:
            allowed_features.add(feature)

    selected_feature_columns = _select_train_only_features(
        train_df=train_df,
        target_column=target_column,
        allowed_features=allowed_features,
    )

    logger.info(
        "Selected features | target=%s | count=%s | features=%s",
        target_column,
        len(selected_feature_columns),
        selected_feature_columns,
    )

    logger.info(
        "Forced feature check | target=%s | %s",
        target_column,
        {
            feature: feature in selected_feature_columns
            for feature in FORCED_PRODUCTION_FEATURES
        },
    )

    X_train = train_df[selected_feature_columns]
    y_train = train_df[target_column]

    X_val = val_df[selected_feature_columns]
    y_val = val_df[target_column]

    X_test = test_df[selected_feature_columns]
    y_test = test_df[target_column]

    model = get_regression_model(
        model_name=algorithm_name,
        model_params=model_params,
    )

    algo = algorithm_name.lower().strip()
    tuning_info: dict[str, Any] | None = None

    if algo in {"xgb", "xgboost"}:
        best_params, best_val_metrics = tune_xgb_regressor(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_iter=XGB_TUNING_ITERATIONS,
            random_state=random_state,
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        )

        model = get_regression_model(
            model_name=algorithm_name,
            model_params=best_params,
        )

        imputer = model.named_steps["imputer"]
        xgb_model = model.named_steps["model"]

        X_train_full = pd.concat(
            [X_train, X_val],
            axis=0,
        )

        y_train_full = pd.concat(
            [y_train, y_val],
            axis=0,
        )

        X_train_full_imp = imputer.fit_transform(
            X_train_full
        )

        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        xgb_model.fit(
            X_train_full_imp,
            y_train_full,
            verbose=False,
        )

        model.named_steps["model"] = xgb_model

        tuning_info = {
            "tuned": True,
            "best_params": best_params,
            "best_validation_metrics_during_tuning": best_val_metrics,
            "tuning_iterations": XGB_TUNING_ITERATIONS,
            "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
        }

        y_val_pred = pd.Series(
            model.named_steps["model"].predict(X_val_imp),
            index=y_val.index,
        )

        y_test_pred = pd.Series(
            model.named_steps["model"].predict(X_test_imp),
            index=y_test.index,
        )

        validation_metrics = best_val_metrics

    elif algo in {"lgbm", "lightgbm"}:
        imputer = model.named_steps["imputer"]
        lgbm_model = model.named_steps["model"]

        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        lgbm_model.fit(
            X_train_imp,
            y_train,
            eval_set=[(X_val_imp, y_val)],
            eval_metric="l2",
        )

        model.named_steps["model"] = lgbm_model

        y_val_pred = pd.Series(
            model.predict(X_val),
            index=y_val.index,
        )

        y_test_pred = pd.Series(
            model.predict(X_test),
            index=y_test.index,
        )

        validation_metrics = build_regression_metrics(
            y_val,
            y_val_pred,
        )

    else:
        model.fit(X_train, y_train)

        y_val_pred = pd.Series(
            model.predict(X_val),
            index=y_val.index,
        )

        y_test_pred = pd.Series(
            model.predict(X_test),
            index=y_test.index,
        )

        validation_metrics = build_regression_metrics(
            y_val,
            y_val_pred,
        )

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model.named_steps["feature_names_"] = selected_feature_columns

    test_metrics = build_regression_metrics(
        y_test,
        y_test_pred,
    )

    metrics = _build_metrics(
        split_result=split_result,
        use_time_based_split=use_time_based_split,
        selected_feature_columns=selected_feature_columns,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        time_column=time_column,
        tuning_info=tuning_info,
    )

    artifact_dir = ""

    if save_artifacts:
        feature_importance_df = _extract_feature_importance_if_supported(
            model=model,
            algorithm_name=algorithm_name,
            selected_feature_columns=selected_feature_columns,
        )

        artifact_dir_path = save_full_artifact_bundle(
            artifacts_root=artifacts_root,
            model_name=artifact_model_name,
            model=model,
            feature_names=selected_feature_columns,
            metrics=metrics,
            target_name=target_column,
            algorithm=algorithm_name,
            preprocessor=preprocessor,
            feature_importance_df=feature_importance_df,
            notes=notes,
            extra_metadata={
                "pipeline_type": "single_target_regression",
                "split_type": (
                    "time_based"
                    if use_time_based_split
                    else "random"
                ),
                "data_period": metrics.get("data_period"),
                "forced_production_features": FORCED_PRODUCTION_FEATURES,
            },
        )

        artifact_dir = str(artifact_dir_path)

        logger.info(
            "Training completed | target=%s | algorithm=%s | artifacts=%s",
            target_column,
            algorithm_name,
            artifact_dir,
        )
    else:
        logger.info(
            "Benchmark completed without artifact saving | target=%s | algorithm=%s | split_type=%s",
            target_column,
            algorithm_name,
            metrics["split_type"],
        )

    return TrainingPipelineResult(
        algorithm_name=algorithm_name,
        artifact_model_name=artifact_model_name,
        target_name=target_column,
        metrics=metrics,
        artifact_dir=artifact_dir,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
    )