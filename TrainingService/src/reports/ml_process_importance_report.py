from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    ARTIFACTS_DIR,
    ML_PROCESS_IMPORTANCE_DIR,
    PROCESS_EXPLAINABILITY_DIR,
)

from utils.io_utils import safe_file_name, write_dataframe_reports

from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


PROCESS_IMPACT_REQUIRED_COLUMNS = {
    "target",
    "process_group",
    "feature",
    "impact_direction",
    "correlation_with_target",
    "absolute_correlation",
    "missing_ratio",
    "valid_pair_count",
}

PROCESS_IMPACT_FILE_NAME = (
    "all_targets_process_impact.csv"
)

ML_IMPORTANCE_FILE_PATTERN = (
    "feature_importance.csv"
)

NO_IMPORTANCE_NOTE = (
    "No ML feature importance files found yet."
)


def _find_feature_importance_files(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> list[Path]:
    artifacts_path = Path(artifacts_dir)

    if not artifacts_path.exists():
        logger.warning(
            "Artifacts directory not found: %s",
            artifacts_path,
        )
        return []

    return list(
        artifacts_path.rglob(
            ML_IMPORTANCE_FILE_PATTERN
        )
    )


def _read_feature_importance_file(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if (
        "feature" not in df.columns
        or "importance" not in df.columns
    ):
        logger.warning(
            "Skipping invalid feature importance file: %s",
            path,
        )
        return pd.DataFrame()

    artifact_dir = path.parent
    metadata_path = artifact_dir / "metadata.json"

    target_name = None
    algorithm_name = None
    artifact_model_name = artifact_dir.name

    if metadata_path.exists():
        metadata = pd.read_json(
            metadata_path,
            typ="series",
        )

        target_name = metadata.get("target_name")
        algorithm_name = metadata.get("algorithm")

    output_df = df.copy()

    output_df["artifact_model_name"] = (
        artifact_model_name
    )

    output_df["target"] = target_name
    output_df["algorithm"] = algorithm_name
    output_df["artifact_dir"] = str(artifact_dir)

    return output_df


@pipeline_step("Load all feature importances")
def load_all_feature_importances(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in _find_feature_importance_files(
        artifacts_dir
    ):
        importance_df = _read_feature_importance_file(
            path
        )

        if not importance_df.empty:
            frames.append(importance_df)

    if not frames:
        logger.warning(
            "No feature importance files found."
        )
        return pd.DataFrame()

    combined_df = (
        pd.concat(frames, axis=0)
        .reset_index(drop=True)
    )

    combined_df["importance"] = pd.to_numeric(
        combined_df["importance"],
        errors="coerce",
    )

    logger.info(
        "Loaded feature importance rows: %s",
        len(combined_df),
    )

    return combined_df


def load_process_impact_report(
    process_report_path: str | Path = (
        PROCESS_EXPLAINABILITY_DIR
        / PROCESS_IMPACT_FILE_NAME
    ),
) -> pd.DataFrame:
    report_path = Path(process_report_path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Process impact report not found: {report_path}. "
            "Run save_process_impact_reports() first."
        )

    report_df = pd.read_csv(report_path)

    missing = (
        PROCESS_IMPACT_REQUIRED_COLUMNS
        - set(report_df.columns)
    )

    if missing:
        raise ValueError(
            "Process impact report is missing "
            f"required columns: {sorted(missing)}"
        )

    return report_df


@pipeline_step("Build ML process importance report")
def build_ml_process_importance_report(
    *,
    process_impact_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> pd.DataFrame:
    if process_impact_df.empty:
        logger.warning(
            "Empty process impact dataframe."
        )
        return pd.DataFrame()

    if feature_importance_df.empty:
        process_only_df = process_impact_df.copy()

        process_only_df["algorithm"] = None
        process_only_df["artifact_model_name"] = None
        process_only_df["ml_importance"] = None
        process_only_df["artifact_dir"] = None
        process_only_df["importance_available"] = False
        process_only_df["combined_rank_score"] = None
        process_only_df["note"] = NO_IMPORTANCE_NOTE

        return process_only_df

    importance_df = (
        feature_importance_df.copy()
        .rename(
            columns={
                "importance": "ml_importance"
            }
        )
    )

    merged_df = process_impact_df.merge(
        importance_df[
            [
                "target",
                "feature",
                "algorithm",
                "artifact_model_name",
                "ml_importance",
                "artifact_dir",
            ]
        ],
        on=["target", "feature"],
        how="left",
    )

    merged_df["ml_importance"] = pd.to_numeric(
        merged_df["ml_importance"],
        errors="coerce",
    )

    merged_df["importance_available"] = (
        merged_df["ml_importance"].notna()
    )

    merged_df["combined_rank_score"] = (
        merged_df["absolute_correlation"]
        .fillna(0)
        + merged_df["ml_importance"]
        .fillna(0)
    )

    merged_df = (
        merged_df
        .sort_values(
            by=[
                "target",
                "algorithm",
                "combined_rank_score",
                "absolute_correlation",
                "ml_importance",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    logger.info(
        "ML process importance rows generated: %s",
        len(merged_df),
    )

    return merged_df


@pipeline_step("Save ML process importance report")
def save_ml_process_importance_report(
    *,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    process_report_path: str | Path = (
        PROCESS_EXPLAINABILITY_DIR
        / PROCESS_IMPACT_FILE_NAME
    ),
    output_dir: str | Path = (
        ML_PROCESS_IMPORTANCE_DIR
    ),
) -> pd.DataFrame:
    process_impact_df = (
        load_process_impact_report(
            process_report_path
        )
    )

    feature_importance_df = (
        load_all_feature_importances(
            artifacts_dir
        )
    )

    report_df = (
        build_ml_process_importance_report(
            process_impact_df=process_impact_df,
            feature_importance_df=feature_importance_df,
        )
    )

    write_dataframe_reports(
        df=report_df,
        output_dir=output_dir,
        base_name="all_targets_ml_process_importance",
    )

    for target in (
        report_df["target"]
        .dropna()
        .unique()
    ):
        target_df = report_df[
            report_df["target"] == target
        ].copy()

        write_dataframe_reports(
            df=target_df,
            output_dir=output_dir,
            base_name=safe_file_name(
                f"{target}_ml_process_importance"
            ),
        )

    logger.info(
        "ML process importance reports saved | rows=%s | targets=%s",
        len(report_df),
        report_df["target"].nunique()
        if not report_df.empty
        else 0,
    )

    return report_df