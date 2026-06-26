# src/data/validator.py

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from config.columns import (
    # REQUIRED_COLUMNS,
    MODEL_FEATURES,
    ALL_TARGETS,
    PRIMARY_TARGETs,
    ID_COLUMNS,
    TIME_COLUMN,
    percentage_attributes,
    positive_cols,
)
from config.settings import (
    VALIDATION_REPORT_PATH,
    MAX_MISSING_RATIO_PER_FEATURE,
    DROP_ROWS_WITH_MISSING_PRIMARY_TARGETS,
)
from utils.decorators import pipeline_step


def get_duplicate_columns(df: pd.DataFrame) -> list[str]:
    return df.columns[df.columns.duplicated()].tolist()


def parse_time_column(df: pd.DataFrame) -> pd.Series:
    if TIME_COLUMN not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index)
    return pd.to_datetime(df[TIME_COLUMN], errors="coerce", dayfirst=True)


def compute_missing_ratios(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {
        col: round(float(df[col].isna().mean()), 4)
        for col in columns
        if col in df.columns
    }


def coerce_numeric_report(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}

    for col in columns:
        if col not in df.columns:
            continue

        raw_non_null = int(df[col].notna().sum())
        coerced = pd.to_numeric(df[col], errors="coerce")
        numeric_non_null = int(coerced.notna().sum())

        report[col] = {
            "raw_non_null": raw_non_null,
            "numeric_non_null": numeric_non_null,
            "non_numeric_count": raw_non_null - numeric_non_null,
        }

    return report


def percentage_range_violations(df: pd.DataFrame) -> dict[str, int]:
    violations = {}
    for col in percentage_attributes:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        count = int(((series < 0) | (series > 100)).sum())
        violations[col] = count
    return violations


def positive_value_violations(df: pd.DataFrame) -> dict[str, int]:
    violations = {}
    for col in positive_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        count = int((series < 0).sum())
        violations[col] = count
    return violations


def detect_constant_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    constant_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            constant_cols.append(col)
    return constant_cols


def build_validation_report(df: pd.DataFrame) -> dict[str, Any]:
    # STRICT required columns (must exist)
    strict_required_columns = ID_COLUMNS + [TIME_COLUMN] + PRIMARY_TARGETs
    missing_required_columns = sorted(list(set(strict_required_columns) - set(df.columns)))

    # OPTIONAL features (nice to have)
    missing_optional_features = sorted(list(set(MODEL_FEATURES) - set(df.columns)))
    duplicate_columns = get_duplicate_columns(df)

    parsed_time = parse_time_column(df)
    missing_feature_ratios = compute_missing_ratios(df, MODEL_FEATURES)
    missing_target_ratios = compute_missing_ratios(df, ALL_TARGETS)

    id_duplicate_count = 0
    if ID_COLUMNS and ID_COLUMNS[0] in df.columns:
        id_duplicate_count = int(df[ID_COLUMNS[0]].duplicated().sum())

    high_missing_feature_columns = sorted([
        col for col, ratio in missing_feature_ratios.items()
        if ratio > MAX_MISSING_RATIO_PER_FEATURE
    ])

    primary_target_missing_count = {}
    for target in PRIMARY_TARGETs:
        if target in df.columns:
            primary_target_missing_count[target] = int(df[target].isna().sum())
        else:
            primary_target_missing_count[target] = None

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "schema_checks": {
            "missing_required_columns": missing_required_columns,
            "missing_optional_features": missing_optional_features,
            "duplicate_column_names": duplicate_columns,
        },
        "time_column": {
            "time_column_name": TIME_COLUMN,
            "parsed_non_null_count": int(parsed_time.notna().sum()),
            "parsed_null_count": int(parsed_time.isna().sum()),
        },
        "id_checks": {
            "id_columns": ID_COLUMNS,
            "duplicate_id_count": id_duplicate_count,
        },
        "missingness": {
            "primary_targets": PRIMARY_TARGETs,
            "primary_target_missing_count": primary_target_missing_count,
            "feature_missing_ratios": missing_feature_ratios,
            "target_missing_ratios": missing_target_ratios,
            "high_missing_feature_columns": high_missing_feature_columns,
        },
        "numeric_conversion": {
            "features": coerce_numeric_report(df, MODEL_FEATURES),
            "targets": coerce_numeric_report(df, ALL_TARGETS),
        },
        "domain_validation": {
            "percentage_range_violations": percentage_range_violations(df),
            "negative_value_violations": positive_value_violations(df),
            "constant_feature_columns": detect_constant_columns(df, MODEL_FEATURES),
        },
        "cleaning_policy": {
            "drop_rows_with_missing_primary_target": DROP_ROWS_WITH_MISSING_PRIMARY_TARGETS,
            "max_missing_ratio_per_feature": MAX_MISSING_RATIO_PER_FEATURE,
        },
    }

    return report


def save_validation_report(report: dict[str, Any]) -> None:
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


@pipeline_step("Validate dataframe")
def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    report = build_validation_report(df)
    save_validation_report(report)
    return report