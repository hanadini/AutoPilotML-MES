from __future__ import annotations

from typing import Iterable

import pandas as pd

from config.columns import TIME_COLUMN
from config.settings import TIME_CV_N_FOLDS
from models.cv_runner import run_model_cv_summary
from models.time_cv import TimeFoldResult
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def _log_cv_summary(
    target: str,
    model_kind: str,
    fold_results: list[TimeFoldResult],
    summary: dict[str, float],
) -> None:
    logger.info(
        "[CV] Target=%s | Model=%s",
        target,
        model_kind,
    )

    for fold in fold_results:
        logger.info(
            (
                "Fold %s | "
                "train=(%s -> %s) | "
                "val=(%s -> %s) | "
                "R2=%.4f | RMSE=%.4f | MAE=%.4f"
            ),
            fold.fold_index,
            fold.train_start,
            fold.train_end,
            fold.val_start,
            fold.val_end,
            fold.r2,
            fold.rmse,
            fold.mae,
        )

    logger.info(
        (
            "CV Summary | "
            "Mean R2=%.4f | "
            "Std R2=%.4f | "
            "Mean RMSE=%.4f | "
            "Mean MAE=%.4f | "
            "Folds=%s"
        ),
        summary["mean_r2"],
        summary["std_r2"],
        summary["mean_rmse"],
        summary["mean_mae"],
        summary["fold_count"],
    )


@pipeline_step("Run cross-validation for target")
def run_cv_for_target(
    *,
    df: pd.DataFrame,
    candidate_features: list[str],
    target: str,
    model_kinds: Iterable[str],
    n_folds: int = TIME_CV_N_FOLDS,
    log_summary: bool = True,
) -> dict[str, tuple[list[TimeFoldResult], dict[str, float]]]:
    cv_results: dict[
        str,
        tuple[list[TimeFoldResult], dict[str, float]],
    ] = {}

    for model_kind in model_kinds:
        fold_results, summary = run_model_cv_summary(
            df=df,
            time_column=TIME_COLUMN,
            candidate_feature_columns=candidate_features,
            target_column=target,
            model_kind=model_kind,
            n_folds=n_folds,
        )

        cv_results[model_kind] = (
            fold_results,
            summary,
        )

        if log_summary:
            _log_cv_summary(
                target,
                model_kind,
                fold_results,
                summary,
            )

    return cv_results