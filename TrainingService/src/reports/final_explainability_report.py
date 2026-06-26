from __future__ import annotations

from pathlib import Path

import pandas as pd



from config.columns import REQUIRED_ML_COLUMNS, REQUIRED_SHAP_COLUMNS, FINAL_REPORT_SORT_COLUMNS, \
    FINAL_REPORT_SORT_ASCENDING

from config.settings import ML_PROCESS_IMPORTANCE_DIR, ML_PROCESS_IMPORTANCE_FILE_NAME, SHAP_EXPLAINABILITY_DIR, \
    SHAP_IMPORTANCE_FILE_NAME, FINAL_EXPLAINABILITY_DIR, FINAL_EXPLAINABILITY_FILE_NAME
from utils.decorators import pipeline_step

from utils.io_utils import write_dataframe_reports, safe_file_name
from utils.logging_utils import get_logger


logger = get_logger(__name__)





def load_ml_process_importance(
    path: str | Path = (
        ML_PROCESS_IMPORTANCE_DIR
        / ML_PROCESS_IMPORTANCE_FILE_NAME
    ),
) -> pd.DataFrame:
    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"ML process importance report not found: {report_path}"
        )

    return pd.read_csv(report_path)


def load_shap_importance(
    path: str | Path = (
        SHAP_EXPLAINABILITY_DIR
        / SHAP_IMPORTANCE_FILE_NAME
    ),
) -> pd.DataFrame:
    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"SHAP importance report not found: {report_path}"
        )

    return pd.read_csv(report_path)


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    report_name: str,
) -> None:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{report_name} missing columns: {sorted(missing)}"
        )


def _coerce_ml_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = df.copy()

    numeric_columns = [
        "correlation_with_target",
        "absolute_correlation",
        "ml_importance",
        "missing_ratio",
    ]

    for column in numeric_columns:
        output_df[column] = pd.to_numeric(
            output_df[column],
            errors="coerce",
        )

    return output_df


def _coerce_shap_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = df.copy()

    output_df["mean_abs_shap"] = pd.to_numeric(
        output_df["mean_abs_shap"],
        errors="coerce",
    )

    output_df["shap_rank"] = pd.to_numeric(
        output_df["shap_rank"],
        errors="coerce",
    )

    return output_df


def _build_final_explainability_score(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df["absolute_correlation"].fillna(0)
        + df["ml_importance"].fillna(0)
        + df["mean_abs_shap"].fillna(0)
    )


def _build_final_rank(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        df.groupby(
            ["target", "algorithm"]
        )
        .cumcount()
        .add(1)
    )


def _ordered_final_columns(
    df: pd.DataFrame,
) -> list[str]:
    ordered_columns = [
        "final_rank",
        "target",
        "algorithm",
        "artifact_model_name",
        "process_group",
        "feature",
        "impact_direction",
        "correlation_with_target",
        "absolute_correlation",
        "ml_importance",
        "shap_rank",
        "mean_abs_shap",
        "final_explainability_score",
        "missing_ratio",
        "valid_pair_count",
        "shap_available",
        "importance_available",
        "artifact_dir",
        "note",
    ]

    existing_ordered_columns = [
        column
        for column in ordered_columns
        if column in df.columns
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in existing_ordered_columns
    ]

    return (
        existing_ordered_columns
        + remaining_columns
    )


@pipeline_step("Build final explainability report")
def build_final_explainability_report(
    *,
    ml_process_df: pd.DataFrame,
    shap_df: pd.DataFrame,
) -> pd.DataFrame:
    _validate_required_columns(
        ml_process_df,
        REQUIRED_ML_COLUMNS,
        "ML process report",
    )

    _validate_required_columns(
        shap_df,
        REQUIRED_SHAP_COLUMNS,
        "SHAP report",
    )

    ml_df = _coerce_ml_numeric_columns(
        ml_process_df
    )

    shap_work_df = _coerce_shap_numeric_columns(
        shap_df
    )

    final_df = ml_df.merge(
        shap_work_df[
            [
                "target",
                "algorithm",
                "artifact_model_name",
                "feature",
                "shap_rank",
                "mean_abs_shap",
            ]
        ],
        on=[
            "target",
            "algorithm",
            "artifact_model_name",
            "feature",
        ],
        how="left",
    )

    final_df["shap_available"] = (
        final_df["mean_abs_shap"].notna()
    )

    final_df["final_explainability_score"] = (
        _build_final_explainability_score(
            final_df
        )
    )

    final_df = (
        final_df
        .sort_values(
            by=FINAL_REPORT_SORT_COLUMNS,
            ascending=FINAL_REPORT_SORT_ASCENDING,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    final_df["final_rank"] = (
        _build_final_rank(final_df)
    )

    final_df = final_df[
        _ordered_final_columns(final_df)
    ]

    logger.info(
        "Final explainability rows generated: %s",
        len(final_df),
    )

    return final_df


@pipeline_step("Save final explainability report")
def save_final_explainability_report(
    *,
    ml_process_path: str | Path = (
        ML_PROCESS_IMPORTANCE_DIR
        / ML_PROCESS_IMPORTANCE_FILE_NAME
    ),
    shap_path: str | Path = (
        SHAP_EXPLAINABILITY_DIR
        / SHAP_IMPORTANCE_FILE_NAME
    ),
    output_dir: str | Path = (
        FINAL_EXPLAINABILITY_DIR
    ),
) -> pd.DataFrame:
    ml_process_df = load_ml_process_importance(
        ml_process_path
    )

    shap_df = load_shap_importance(
        shap_path
    )

    final_df = (
        build_final_explainability_report(
            ml_process_df=ml_process_df,
            shap_df=shap_df,
        )
    )

    write_dataframe_reports(
        df=final_df,
        output_dir=output_dir,
        base_name=FINAL_EXPLAINABILITY_FILE_NAME,
    )

    for target in (
        final_df["target"]
        .dropna()
        .unique()
    ):
        target_df = final_df[
            final_df["target"] == target
        ].copy()

        write_dataframe_reports(
            df=target_df,
            output_dir=output_dir,
            base_name=safe_file_name(
                f"{target}_final_explainability"
            ),
        )

    logger.info(
        "Final explainability reports saved | rows=%s | targets=%s",
        len(final_df),
        final_df["target"].nunique()
        if not final_df.empty
        else 0,
    )

    return final_df