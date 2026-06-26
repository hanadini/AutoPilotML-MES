from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    ARTIFACTS_DIR,
    ENABLE_RANDOM_SPLIT_BENCHMARK,
    ENABLE_TIME_CV_BENCHMARK,
    ENABLE_TIME_SPLIT_BENCHMARK,
    EVALUATION_COMPARISON_DIR,
)
from models.cv_orchestrator import run_cv_for_target
from pipeline.training_pipeline import run_single_target_regression_pipeline
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


TIME_CV_SUPPORTED_MODELS = {
    "rf",
    "xgb",
    "ensemble_xgb_rf",
}


def _stability_label(gap: float | None) -> str:
    if gap is None:
        return "not_available"

    if gap < 0.05:
        return "stable"

    if gap < 0.15:
        return "moderate_drift"

    return "high_drift"


def _recommendation(
    *,
    random_r2: float | None,
    time_split_r2: float | None,
    time_cv_mean_r2: float | None,
    gap: float | None,
) -> str:
    if random_r2 is None and time_split_r2 is None and time_cv_mean_r2 is None:
        return "No evaluation result available."

    if gap is None:
        return "Use available benchmark results with caution."

    if gap < 0.05:
        return "Model behaviour is stable across random and time-based evaluation."

    if gap < 0.15:
        return "Moderate temporal drift detected. Prefer time-based metrics for production selection."

    return "High temporal drift detected. Investigate process changes, data drift, or feature stability before deployment."


def _extract_cv_summary(
    cv_results: dict[str, Any],
    algorithm_name: str,
) -> tuple[float | None, float | None, int | None]:
    if algorithm_name not in cv_results:
        return None, None, None

    _, summary = cv_results[algorithm_name]

    return (
        summary.get("mean_r2"),
        summary.get("std_r2"),
        summary.get("fold_count"),
    )


def _run_benchmark_model(
    *,
    df: pd.DataFrame,
    candidate_features: list[str],
    target: str,
    algorithm_name: str,
    artifact_model_name: str,
    model_params: dict | None,
    time_column: str,
    use_time_based_split: bool,
    notes: str,
) -> Any:
    return run_single_target_regression_pipeline(
        df=df,
        candidate_feature_columns=candidate_features,
        target_column=target,
        algorithm_name=algorithm_name,
        artifact_model_name=artifact_model_name,
        model_params=model_params,
        artifacts_root=ARTIFACTS_DIR,
        time_column=time_column,
        use_time_based_split=use_time_based_split,
        notes=notes,
        save_artifacts=False,
    )


def _save_evaluation_comparison_report(
    comparison_df: pd.DataFrame,
) -> tuple[Path, Path]:
    output_dir = Path(EVALUATION_COMPARISON_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "evaluation_strategy_comparison.csv"
    xlsx_path = output_dir / "evaluation_strategy_comparison.xlsx"

    comparison_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_excel(
        xlsx_path,
        index=False,
    )

    return csv_path, xlsx_path


@pipeline_step("Run evaluation strategy comparison")
def run_evaluation_comparison(
    *,
    df: pd.DataFrame,
    candidate_features: list[str],
    targets: list[str],
    algorithm_configs: list[Any],
    time_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for target in targets:
        logger.info(
            "Evaluation comparison started | target=%s",
            target,
        )

        cv_results: dict[str, Any] = {}

        if ENABLE_TIME_CV_BENCHMARK:
            try:
                cv_results = run_cv_for_target(
                    df=df,
                    candidate_features=candidate_features,
                    target=target,
                    model_kinds=TIME_CV_SUPPORTED_MODELS,
                )
            except Exception as exc:
                logger.warning(
                    "Time CV benchmark failed | target=%s | error=%s",
                    target,
                    exc,
                )

        for algo_cfg in algorithm_configs:
            algorithm_name = algo_cfg.algorithm_name
            model_params = algo_cfg.model_params

            random_result = None
            time_result = None
            error_message = ""

            logger.info(
                "Evaluation comparison | target=%s | algorithm=%s",
                target,
                algorithm_name,
            )

            try:
                if ENABLE_RANDOM_SPLIT_BENCHMARK:
                    random_result = _run_benchmark_model(
                        df=df,
                        candidate_features=candidate_features,
                        target=target,
                        algorithm_name=algorithm_name,
                        artifact_model_name=(
                            f"{target}_{algorithm_name}_random_benchmark"
                        ),
                        model_params=model_params,
                        time_column=time_column,
                        use_time_based_split=False,
                        notes="Random split benchmark. No artifact saved.",
                    )

                if ENABLE_TIME_SPLIT_BENCHMARK:
                    time_result = _run_benchmark_model(
                        df=df,
                        candidate_features=candidate_features,
                        target=target,
                        algorithm_name=algorithm_name,
                        artifact_model_name=(
                            f"{target}_{algorithm_name}_time_benchmark"
                        ),
                        model_params=model_params,
                        time_column=time_column,
                        use_time_based_split=True,
                        notes="Time split benchmark. No artifact saved.",
                    )

            except Exception as exc:
                error_message = str(exc)
                logger.warning(
                    "Evaluation benchmark failed | target=%s | algorithm=%s | error=%s",
                    target,
                    algorithm_name,
                    exc,
                )

            random_r2 = (
                random_result.metrics["test"]["r2"]
                if random_result is not None
                else None
            )

            random_rmse = (
                random_result.metrics["test"]["rmse"]
                if random_result is not None
                else None
            )

            random_mae = (
                random_result.metrics["test"]["mae"]
                if random_result is not None
                else None
            )

            time_split_r2 = (
                time_result.metrics["test"]["r2"]
                if time_result is not None
                else None
            )

            time_split_rmse = (
                time_result.metrics["test"]["rmse"]
                if time_result is not None
                else None
            )

            time_split_mae = (
                time_result.metrics["test"]["mae"]
                if time_result is not None
                else None
            )

            gap = (
                random_r2 - time_split_r2
                if random_r2 is not None and time_split_r2 is not None
                else None
            )

            time_cv_mean_r2, time_cv_std_r2, time_cv_fold_count = (
                _extract_cv_summary(
                    cv_results=cv_results,
                    algorithm_name=algorithm_name,
                )
            )

            rows.append(
                {
                    "target": target,
                    "algorithm": algorithm_name,
                    "random_test_r2": random_r2,
                    "random_test_rmse": random_rmse,
                    "random_test_mae": random_mae,
                    "time_split_test_r2": time_split_r2,
                    "time_split_test_rmse": time_split_rmse,
                    "time_split_test_mae": time_split_mae,
                    "time_cv_mean_r2": time_cv_mean_r2,
                    "time_cv_std_r2": time_cv_std_r2,
                    "time_cv_fold_count": time_cv_fold_count,
                    "random_vs_time_r2_gap": gap,
                    "stability_label": _stability_label(gap),
                    "recommendation": _recommendation(
                        random_r2=random_r2,
                        time_split_r2=time_split_r2,
                        time_cv_mean_r2=time_cv_mean_r2,
                        gap=gap,
                    ),
                    "time_cv_supported": algorithm_name in TIME_CV_SUPPORTED_MODELS,
                    "error": error_message,
                }
            )

    comparison_df = pd.DataFrame(rows)

    csv_path, xlsx_path = _save_evaluation_comparison_report(
        comparison_df,
    )

    logger.info(
        "Evaluation strategy comparison saved | csv=%s | xlsx=%s",
        csv_path,
        xlsx_path,
    )

    return comparison_df