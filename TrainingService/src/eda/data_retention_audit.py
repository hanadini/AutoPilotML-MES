from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config.settings as settings

from config.columns import PRIMARY_TARGETs, MODEL_FEATURES, TIME_COLUMN
from config.settings import (
    TARGET_AVAILABILITY_REPORT_PATH,
    TARGET_COMBINATION_REPORT_PATH,
    RETENTION_SUMMARY_PATH,
    RETENTION_REPORT_PATH,
)

print(settings.__file__)


@dataclass
class RetentionStep:
    step_name: str
    rows_before: int
    rows_after: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def percent_removed(self) -> float:
        if self.rows_before == 0:
            return 0.0
        return round(100 * self.rows_removed / self.rows_before, 2)


class DataRetentionAuditor:
    def __init__(self) -> None:
        self.steps: list[RetentionStep] = []

    def add_step(
        self,
        step_name: str,
        rows_before: int,
        rows_after: int,
    ) -> None:
        self.steps.append(
            RetentionStep(
                step_name=step_name,
                rows_before=rows_before,
                rows_after=rows_after,
            )
        )

    def build_report(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "step": step.step_name,
                    "rows_before": step.rows_before,
                    "rows_after": step.rows_after,
                    "rows_removed": step.rows_removed,
                    "percent_removed": step.percent_removed,
                }
                for step in self.steps
            ]
        )

    def save(self) -> Path:
        RETENTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        report_df = self.build_report()
        report_df.to_csv(
            RETENTION_REPORT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        return RETENTION_REPORT_PATH


def build_target_availability_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for target in PRIMARY_TARGETs:
        if target not in df.columns:
            rows.append(
                {
                    "target": target,
                    "available_rows": 0,
                    "missing_rows": len(df),
                    "availability_percent": 0.0,
                    "status": "missing_column",
                }
            )
            continue

        target_series = pd.to_numeric(df[target], errors="coerce")
        available_rows = int(target_series.notna().sum())
        missing_rows = int(target_series.isna().sum())

        rows.append(
            {
                "target": target,
                "available_rows": available_rows,
                "missing_rows": missing_rows,
                "availability_percent": round(
                    100 * available_rows / len(df),
                    2,
                ) if len(df) else 0.0,
                "status": "available_column",
            }
        )

    return pd.DataFrame(rows)


def build_target_combination_report(df: pd.DataFrame) -> pd.DataFrame:
    targets = [target for target in PRIMARY_TARGETs if target in df.columns]

    if not targets:
        return pd.DataFrame(
            columns=[
                "minimum_targets_available",
                "rows",
                "percent",
            ]
        )

    numeric_targets = df[targets].apply(pd.to_numeric, errors="coerce")
    available_target_count = numeric_targets.notna().sum(axis=1)

    rows = []

    for n in range(1, len(targets) + 1):
        count = int((available_target_count >= n).sum())

        rows.append(
            {
                "minimum_targets_available": n,
                "rows": count,
                "percent": round(100 * count / len(df), 2) if len(df) else 0.0,
            }
        )

    exact_all_count = int((available_target_count == len(targets)).sum())

    rows.append(
        {
            "minimum_targets_available": "all_primary_targets",
            "rows": exact_all_count,
            "percent": round(100 * exact_all_count / len(df), 2) if len(df) else 0.0,
        }
    )

    return pd.DataFrame(rows)


def build_feature_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in MODEL_FEATURES:
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "available_rows": 0,
                    "missing_rows": len(df),
                    "missing_percent": 100.0,
                    "status": "missing_column",
                }
            )
            continue

        series = pd.to_numeric(df[feature], errors="coerce")
        missing_rows = int(series.isna().sum())
        available_rows = int(series.notna().sum())

        rows.append(
            {
                "feature": feature,
                "available_rows": available_rows,
                "missing_rows": missing_rows,
                "missing_percent": round(100 * missing_rows / len(df), 2)
                if len(df)
                else 0.0,
                "status": "available_column",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by="missing_percent", ascending=False)
        .reset_index(drop=True)
    )


def build_retention_summary_text(
    *,
    raw_rows: int,
    final_rows: int,
    target_availability_df: pd.DataFrame,
    target_combination_df: pd.DataFrame,
) -> str:
    rows_removed = raw_rows - final_rows
    retention_rate = round(100 * final_rows / raw_rows, 2) if raw_rows else 0.0
    data_loss_rate = round(100 * rows_removed / raw_rows, 2) if raw_rows else 0.0

    lines = [
        "MDF1 DATA RETENTION ANALYSIS",
        "=" * 60,
        "",
        f"Initial rows: {raw_rows}",
        f"Final rows before ML: {final_rows}",
        f"Rows removed: {rows_removed}",
        f"Retention rate: {retention_rate}%",
        f"Data loss rate: {data_loss_rate}%",
        "",
        "TARGET AVAILABILITY",
        "-" * 60,
    ]

    for _, row in target_availability_df.iterrows():
        lines.extend(
            [
                f"Target: {row['target']}",
                f"  Available rows: {row['available_rows']}",
                f"  Missing rows: {row['missing_rows']}",
                f"  Availability: {row['availability_percent']}%",
                "",
            ]
        )

    lines.extend(
        [
            "TARGET COMBINATION AVAILABILITY",
            "-" * 60,
        ]
    )

    for _, row in target_combination_df.iterrows():
        lines.append(
            f"Minimum targets available = {row['minimum_targets_available']}: "
            f"{row['rows']} rows ({row['percent']}%)"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 60,
            "The final training dataset is limited mainly by laboratory target availability.",
            "If all primary targets are required simultaneously, only rows with complete target coverage can be used.",
            "For target-specific training, more rows may be usable by training each target independently.",
        ]
    )

    return "\n".join(lines)


def save_pre_ml_data_retention_reports(
    *,
    raw_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> dict[str, Path]:
    target_availability_df = build_target_availability_report(raw_df)
    target_combination_df = build_target_combination_report(raw_df)
    feature_missingness_df = build_feature_missingness_report(raw_df)

    TARGET_AVAILABILITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_COMBINATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETENTION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    target_availability_df.to_csv(
        TARGET_AVAILABILITY_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    target_combination_df.to_csv(
        TARGET_COMBINATION_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    feature_missingness_path = (
        TARGET_AVAILABILITY_REPORT_PATH.parent / "feature_missingness_report.csv"
    )

    feature_missingness_df.to_csv(
        feature_missingness_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_text = build_retention_summary_text(
        raw_rows=len(raw_df),
        final_rows=len(final_df),
        target_availability_df=target_availability_df,
        target_combination_df=target_combination_df,
    )

    RETENTION_SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    return {
        "target_availability_report": TARGET_AVAILABILITY_REPORT_PATH,
        "target_combination_report": TARGET_COMBINATION_REPORT_PATH,
        "feature_missingness_report": feature_missingness_path,
        "retention_summary": RETENTION_SUMMARY_PATH,
    }