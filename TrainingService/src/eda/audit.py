from __future__ import annotations

import pandas as pd

from config.columns import MODEL_FEATURES, PRIMARY_TARGETs, TIME_COLUMN
from config.settings import AUDIT_SUMMARY_PATH
from utils.decorators import pipeline_step


def build_audit_text(df: pd.DataFrame) -> str:
    lines: list[str] = []

    lines.append("MDF1 INITIAL DATA AUDIT")
    lines.append("=" * 60)
    lines.append(f"Rows: {df.shape[0]}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append("")

    if TIME_COLUMN in df.columns:
        lines.append(f"Time column: {TIME_COLUMN}")
        lines.append(f"Min time: {df[TIME_COLUMN].min()}")
        lines.append(f"Max time: {df[TIME_COLUMN].max()}")
        lines.append("")

    existing_targets = [
        target
        for target in PRIMARY_TARGETs
        if target in df.columns
    ]

    if existing_targets:
        lines.append("Primary targets summary:")

        for target in existing_targets:
            target_series = pd.to_numeric(
                df[target],
                errors="coerce",
            )

            lines.append(f"  Target: {target}")
            lines.append(
                f"    Non-null count: {target_series.notna().sum()}"
            )
            lines.append(f"    Mean: {target_series.mean():.4f}")
            lines.append(f"    Std: {target_series.std():.4f}")
            lines.append(f"    Min: {target_series.min():.4f}")
            lines.append(f"    Max: {target_series.max():.4f}")
            lines.append("")

    numeric_features = [
        feature
        for feature in MODEL_FEATURES
        if feature in df.columns
    ]

    if existing_targets and numeric_features:
        for target in existing_targets:
            numeric_df = df[numeric_features + [target]].apply(
                pd.to_numeric,
                errors="coerce",
            )

            corr = numeric_df.corr(numeric_only=True)[target]
            corr = corr.drop(
                labels=[target],
                errors="ignore",
            ).dropna()

            top_corr = (
                corr.abs()
                .sort_values(ascending=False)
                .head(15)
            )

            lines.append(
                f"Top 15 absolute correlations with target: {target}"
            )

            for col in top_corr.index:
                lines.append(f"  - {col}: {corr[col]:.4f}")

            lines.append("")

    return "\n".join(lines)


def save_audit_text(text: str) -> None:
    AUDIT_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(AUDIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)


@pipeline_step("Run initial data audit")
def run_initial_audit(df: pd.DataFrame) -> str:
    text = build_audit_text(df)
    save_audit_text(text)
    return text