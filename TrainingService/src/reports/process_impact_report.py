from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.columns import (
    ALL_TARGETS,
    PROCESS_FEATURE_GROUPS,
)
from config.settings import PROCESS_EXPLAINABILITY_DIR, PROCESS_IMPACT_REQUIRED_MIN_VALID_PAIRS, PROCESS_IMPACT_NOTE
from utils.decorators import pipeline_step
from utils.io_utils import safe_file_name, write_dataframe_reports
from utils.logging_utils import get_logger


logger = get_logger(__name__)



def _build_feature_to_group_map() -> dict[str, str]:
    feature_to_group: dict[str, str] = {}

    for group_name, features in PROCESS_FEATURE_GROUPS.items():
        for feature in features:
            feature_to_group[feature] = group_name

    return feature_to_group


def _safe_numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def _impact_direction(
    correlation: float | None,
) -> str:
    if correlation is None or pd.isna(correlation):
        return "unknown"

    if correlation > 0:
        return "positive"

    if correlation < 0:
        return "negative"

    return "neutral"


@pipeline_step("Build process impact report for target")
def build_process_impact_report_for_target(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """
    Build process impact report for one target.

    This report answers:
    - Which process features are related to this target?
    - Which process group does each feature belong to?
    - Is the relationship positive or negative?
    - How much missing data does each signal have?

    Important:
    This is an explainability / analysis report.
    It does not prove causality.
    """
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found in dataframe."
        )

    feature_to_group = _build_feature_to_group_map()

    process_features = list(feature_to_group.keys())

    existing_features = [
        feature
        for feature in process_features
        if feature in df.columns
    ]

    rows: list[dict[str, Any]] = []

    target_series = _safe_numeric_series(
        df,
        target,
    )

    for feature in existing_features:
        feature_series = _safe_numeric_series(
            df,
            feature,
        )

        valid_pair = pd.concat(
            [feature_series, target_series],
            axis=1,
        ).dropna()

        valid_pair.columns = [feature, target]

        if len(valid_pair) >= PROCESS_IMPACT_REQUIRED_MIN_VALID_PAIRS:
            correlation = float(
                valid_pair[feature].corr(
                    valid_pair[target]
                )
            )
        else:
            correlation = None

        rows.append(
            {
                "target": target,
                "process_group": feature_to_group[feature],
                "feature": feature,
                "available": True,
                "row_count": int(len(df)),
                "non_null_count": int(
                    feature_series.notna().sum()
                ),
                "missing_count": int(
                    feature_series.isna().sum()
                ),
                "missing_ratio": round(
                    float(feature_series.isna().mean()),
                    4,
                ),
                "target_non_null_count": int(
                    target_series.notna().sum()
                ),
                "valid_pair_count": int(len(valid_pair)),
                "correlation_with_target": (
                    round(correlation, 6)
                    if correlation is not None
                    and not pd.isna(correlation)
                    else None
                ),
                "absolute_correlation": (
                    round(abs(correlation), 6)
                    if correlation is not None
                    and not pd.isna(correlation)
                    else None
                ),
                "impact_direction": _impact_direction(
                    correlation
                ),
                "note": PROCESS_IMPACT_NOTE,
            }
        )

    report_df = pd.DataFrame(rows)

    if report_df.empty:
        return report_df

    report_df = (
        report_df
        .sort_values(
            by=[
                "absolute_correlation",
                "valid_pair_count",
            ],
            ascending=[False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    report_df["impact_rank"] = range(
        1,
        len(report_df) + 1,
    )

    ordered_columns = [
        "impact_rank",
        "target",
        "process_group",
        "feature",
        "impact_direction",
        "correlation_with_target",
        "absolute_correlation",
        "valid_pair_count",
        "non_null_count",
        "missing_count",
        "missing_ratio",
        "target_non_null_count",
        "row_count",
        "available",
        "note",
    ]

    return report_df[ordered_columns]


@pipeline_step("Build all process impact reports")
def build_all_process_impact_reports(
    df: pd.DataFrame,
    targets: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    selected_targets = targets or ALL_TARGETS

    reports: dict[str, pd.DataFrame] = {}

    for target in selected_targets:
        if target not in df.columns:
            logger.warning(
                "Skipping missing target column: %s",
                target,
            )
            continue

        reports[target] = (
            build_process_impact_report_for_target(
                df=df,
                target=target,
            )
        )

    return reports


@pipeline_step("Save process impact reports")
def save_process_impact_reports(
    df: pd.DataFrame,
    targets: list[str] | None = None,
    output_dir: str | Path = PROCESS_EXPLAINABILITY_DIR,
) -> pd.DataFrame:
    """
    Save:
    - one report per target
    - one combined report
    """
    reports = build_all_process_impact_reports(
        df=df,
        targets=targets,
    )

    combined_reports: list[pd.DataFrame] = []

    for target, report_df in reports.items():
        if report_df.empty:
            logger.warning(
                "Empty process impact report: %s",
                target,
            )
            continue

        write_dataframe_reports(
            df=report_df,
            output_dir=output_dir,
            base_name=safe_file_name(
                f"{target}_process_impact"
            ),
        )

        combined_reports.append(report_df)

    if combined_reports:
        combined_df = (
            pd.concat(combined_reports, axis=0)
            .reset_index(drop=True)
        )
    else:
        combined_df = pd.DataFrame()

    if not combined_df.empty:
        write_dataframe_reports(
            df=combined_df,
            output_dir=output_dir,
            base_name="all_targets_process_impact",
        )

    logger.info(
        "Process impact reports saved | targets=%s | total_rows=%s",
        len(reports),
        len(combined_df),
    )

    return combined_df