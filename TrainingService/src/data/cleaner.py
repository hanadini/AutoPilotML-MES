from __future__ import annotations

import numpy as np
import pandas as pd

from config.columns import (
    MODEL_FEATURES,
    ALL_TARGETS,
    PRIMARY_TARGETs,
    ID_COLUMNS,
    TIME_COLUMN, FORCED_PRODUCTION_FEATURES,
)
from config.settings import (
    CLEANED_FILE_PATH,
    MAX_MISSING_RATIO_PER_FEATURE, TEXT_NA_VALUES,
)
from utils.decorators import log_dataframe_shape, pipeline_step





def normalize_text_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.replace(list(TEXT_NA_VALUES), np.nan)


def parse_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TIME_COLUMN in df.columns:
        df[TIME_COLUMN] = pd.to_datetime(
            df[TIME_COLUMN],
            errors="coerce",
            dayfirst=True,
        )

    return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        col
        for col in MODEL_FEATURES + ALL_TARGETS
        if col in df.columns
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def drop_rows_without_primary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    existing_targets = [
        col
        for col in PRIMARY_TARGETs
        if col in df.columns
    ]

    if existing_targets:
        df = df[df[existing_targets].notna().all(axis=1)].copy()

    return df


def drop_high_missing_feature_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    protected_features = set(FORCED_PRODUCTION_FEATURES)
    removable_features: list[str] = []

    for col in MODEL_FEATURES:
        if col not in df.columns:
            continue

        if col in protected_features:
            continue

        missing_ratio = df[col].isna().mean()

        if missing_ratio > MAX_MISSING_RATIO_PER_FEATURE:
            removable_features.append(col)

    df = df.drop(columns=removable_features, errors="ignore")

    return df, removable_features


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TIME_COLUMN in df.columns:
        df = df.sort_values(by=TIME_COLUMN, ascending=True)

    return df


def basic_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    subset_cols = [
        col for col in ID_COLUMNS + [TIME_COLUMN]
        if col in df.columns
    ]

    if subset_cols:
        return df.drop_duplicates(subset=subset_cols, keep="first")

    return df.drop_duplicates(keep="first")


def save_cleaned_dataframe(df: pd.DataFrame) -> None:
    CLEANED_FILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_to_save = df.copy()

    object_columns = (
        df_to_save
        .select_dtypes(include=["object"])
        .columns
    )

    for column in object_columns:
        df_to_save[column] = df_to_save[column].apply(
            lambda value:
            str(value).strip()
            if pd.notna(value)
            else value
        )

    df_to_save.to_parquet(
        CLEANED_FILE_PATH,
        index=False,
    )


@pipeline_step("Clean dataframe")
@log_dataframe_shape("Cleaned dataframe")
def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = normalize_text_na(df)
    df = parse_time_column(df)
    df = coerce_numeric_columns(df)
    df = drop_rows_without_primary_target(df)
    df = basic_deduplicate(df)
    df = sort_by_time(df)
    df, dropped_features = drop_high_missing_feature_columns(df)

    save_cleaned_dataframe(df)

    return df, dropped_features