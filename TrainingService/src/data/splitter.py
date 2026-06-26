from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    IID_TEST_SIZE,
    IID_VALID_SIZE,
    RANDOM_STATE,
)
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def _validate_split_sizes(
    test_size: float,
    val_size: float,
) -> None:
    if test_size <= 0 or val_size <= 0:
        raise ValueError("test_size and val_size must be > 0.")

    if (test_size + val_size) >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")


def _to_datetime_series(
    df: pd.DataFrame,
    time_column: str,
) -> pd.Series:
    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' not found in dataframe."
        )

    return pd.to_datetime(
        df[time_column],
        errors="coerce",
    )


def _format_datetime(value) -> str | None:
    if pd.isna(value):
        return None

    return pd.Timestamp(value).isoformat()


def build_split_date_range_report(
    split_result: SplitResult,
    time_column: str,
) -> dict[str, str | int | None]:
    train_dates = _to_datetime_series(
        split_result.train_df,
        time_column,
    )

    val_dates = _to_datetime_series(
        split_result.val_df,
        time_column,
    )

    test_dates = _to_datetime_series(
        split_result.test_df,
        time_column,
    )

    all_dates = pd.concat(
        [train_dates, val_dates, test_dates],
        axis=0,
    ).dropna()

    if all_dates.empty:
        raise ValueError(
            f"No valid datetime values found in '{time_column}'."
        )

    return {
        "time_column": time_column,
        "raw_data_start": _format_datetime(all_dates.min()),
        "raw_data_end": _format_datetime(all_dates.max()),
        "total_rows": int(
            len(split_result.train_df)
            + len(split_result.val_df)
            + len(split_result.test_df)
        ),
        "train_start": _format_datetime(train_dates.min()),
        "train_end": _format_datetime(train_dates.max()),
        "train_rows": int(len(split_result.train_df)),
        "validation_start": _format_datetime(val_dates.min()),
        "validation_end": _format_datetime(val_dates.max()),
        "validation_rows": int(len(split_result.val_df)),
        "test_start": _format_datetime(test_dates.min()),
        "test_end": _format_datetime(test_dates.max()),
        "test_rows": int(len(split_result.test_df)),
    }


def log_split_date_ranges(
    split_result: SplitResult,
    time_column: str,
) -> None:
    report = build_split_date_range_report(
        split_result=split_result,
        time_column=time_column,
    )

    logger.info(
        "Raw data period: %s -> %s | rows=%s",
        report["raw_data_start"],
        report["raw_data_end"],
        report["total_rows"],
    )

    logger.info(
        "Train period: %s -> %s | rows=%s",
        report["train_start"],
        report["train_end"],
        report["train_rows"],
    )

    logger.info(
        "Validation period: %s -> %s | rows=%s",
        report["validation_start"],
        report["validation_end"],
        report["validation_rows"],
    )

    logger.info(
        "Test period: %s -> %s | rows=%s",
        report["test_start"],
        report["test_end"],
        report["test_rows"],
    )


@pipeline_step("Random train/validation/test split")
def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = IID_TEST_SIZE,
    val_size: float = IID_VALID_SIZE,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
    time_column: str | None = None,
) -> SplitResult:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    _validate_split_sizes(
        test_size=test_size,
        val_size=val_size,
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    split_result = SplitResult(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )

    if time_column is not None and time_column in df.columns:
        log_split_date_ranges(
            split_result=split_result,
            time_column=time_column,
        )

    return split_result


@pipeline_step("Time-based train/validation/test split")
def split_time_based(
    df: pd.DataFrame,
    time_column: str,
    test_size: float = IID_TEST_SIZE,
    val_size: float = IID_VALID_SIZE,
) -> SplitResult:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    _validate_split_sizes(
        test_size=test_size,
        val_size=val_size,
    )

    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' not found in dataframe."
        )

    ordered_df = df.copy()

    ordered_df[time_column] = pd.to_datetime(
        ordered_df[time_column],
        errors="coerce",
    )

    invalid_time_count = int(
        ordered_df[time_column].isna().sum()
    )

    if invalid_time_count > 0:
        logger.warning(
            "Dropping %s rows with invalid '%s' values before time-based split.",
            invalid_time_count,
            time_column,
        )

    ordered_df = ordered_df.dropna(
        subset=[time_column],
    )

    if ordered_df.empty:
        raise ValueError(
            f"No valid rows remain after parsing '{time_column}'."
        )

    ordered_df = ordered_df.sort_values(
        by=time_column,
    ).reset_index(drop=True)

    n_rows = len(ordered_df)
    test_count = int(n_rows * test_size)
    val_count = int(n_rows * val_size)
    train_count = n_rows - val_count - test_count

    if train_count <= 0:
        raise ValueError("Not enough rows for time-based split.")

    if val_count <= 0:
        raise ValueError("Not enough rows for validation split.")

    if test_count <= 0:
        raise ValueError("Not enough rows for test split.")

    train_df = ordered_df.iloc[
        :train_count
    ].copy()

    val_df = ordered_df.iloc[
        train_count:train_count + val_count
    ].copy()

    test_df = ordered_df.iloc[
        train_count + val_count:
    ].copy()

    split_result = SplitResult(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
    )

    validate_time_order(
        split_result=split_result,
        time_column=time_column,
    )

    log_split_date_ranges(
        split_result=split_result,
        time_column=time_column,
    )

    return split_result


def validate_time_order(
    split_result: SplitResult,
    time_column: str,
) -> None:
    if split_result.train_df.empty:
        raise ValueError("Train split is empty.")

    if split_result.val_df.empty:
        raise ValueError("Validation split is empty.")

    if split_result.test_df.empty:
        raise ValueError("Test split is empty.")

    train_max = _to_datetime_series(
        split_result.train_df,
        time_column,
    ).max()

    val_min = _to_datetime_series(
        split_result.val_df,
        time_column,
    ).min()

    val_max = _to_datetime_series(
        split_result.val_df,
        time_column,
    ).max()

    test_min = _to_datetime_series(
        split_result.test_df,
        time_column,
    ).min()

    if train_max > val_min:
        raise ValueError("Train overlaps validation.")

    if val_max > test_min:
        raise ValueError("Validation overlaps test.")