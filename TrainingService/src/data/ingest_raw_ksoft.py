from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config.settings import (
    COLUMN_MAPPING_FILE_PATH,
    FLATTENED_FILE_PATH,
    INTERIM_DIR,
    MES_FEATURE_MAPPING_FILE_PATH,
    METADATA_DIR,
    RAW_DIR,
    RAW_FILE_PATTERN,
    SHEET_NAME,
)
from utils.decorators import (
    log_dataframe_shape,
    pipeline_step,
)
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def normalize_mes_feature_name(
    name: str,
) -> str:
    """
    Convert FQTTS human-readable names>>>>into MES-style names.

    Example:
    "Press Fibre Density"
        -> press_fibre_density

    "Belt Speed 1"
        -> belt_speed_1
    """

    name = str(name).strip().lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    return name.strip("_")


def ensure_directories() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_excel_files() -> list[Path]:
    excel_files = sorted(RAW_DIR.glob(RAW_FILE_PATTERN))

    if not excel_files:
        raise FileNotFoundError(
            f"No Excel files found in: {RAW_DIR}"
        )

    return excel_files


@pipeline_step("Load raw Excel sheets")
@log_dataframe_shape("Raw merged dataframe")
def load_raw_sheet() -> pd.DataFrame:
    """
    Load and merge all Excel files from RAW_DIR.
    First file keeps header rows.
    Other files skip the first 2 header rows.
    """
    excel_files = get_raw_excel_files()

    dataframes = []

    for index, excel_file in enumerate(excel_files):
        df = pd.read_excel(
            excel_file,
            sheet_name=SHEET_NAME,
            header=None,
            engine="openpyxl",
        )

        if index > 0:
            df = df.iloc[2:].copy()

        dataframes.append(df)

    return pd.concat(
        dataframes,
        axis=0,
        ignore_index=True,
    )


def extract_human_readable_names(
    raw_df: pd.DataFrame,
) -> list[str]:
    """
    Excel row 1 -> human-readable labels
    pandas index 0
    """
    return [str(col).strip() for col in raw_df.iloc[0].tolist()]


def extract_technical_names(
    raw_df: pd.DataFrame,
) -> list[str]:
    """
    Excel row 2 -> technical FQTTS names
    pandas index 1
    """
    technical_names = [
        str(col).strip()
        for col in raw_df.iloc[1].tolist()
    ]

    cleaned_names = []

    for idx, name in enumerate(technical_names):
        if name.lower() in {"nan", "none", ""}:
            cleaned_names.append(f"unnamed_col_{idx}")
        else:
            cleaned_names.append(name)

    return cleaned_names


def make_unique(names: list[str]) -> list[str]:
    """
    Ensure duplicate column names become unique.
    """
    seen: dict[str, int] = {}
    unique_names: list[str] = []

    for name in names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}__{seen[name]}")

    return unique_names


def build_model_dataframe(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build modeling dataframe using technical names.
    """
    technical_names = extract_technical_names(raw_df)
    technical_names = make_unique(technical_names)

    df = raw_df.iloc[2:].copy()

    df.columns = technical_names

    df.reset_index(
        drop=True,
        inplace=True,
    )

    return df


def build_column_mapping(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build mapping between human-readable
    and technical column names.
    """
    human_names = extract_human_readable_names(raw_df)

    technical_names = extract_technical_names(raw_df)
    technical_names = make_unique(technical_names)

    max_len = max(
        len(human_names),
        len(technical_names),
    )

    if len(human_names) < max_len:
        human_names += [""] * (
            max_len - len(human_names)
        )

    if len(technical_names) < max_len:
        technical_names += [""] * (
            max_len - len(technical_names)
        )

    return pd.DataFrame(
        {
            "human_readable_name": human_names,
            "technical_name": technical_names,
        }
    )


def save_column_mapping(
    mapping_df: pd.DataFrame,
) -> None:
    mapping_df.to_csv(
        COLUMN_MAPPING_FILE_PATH,
        index=False,
        encoding="utf-8-sig",
    )



def save_mes_feature_mapping(
    mapping_df: pd.DataFrame,
) -> None:
    """
    Save runtime MES-to-ML feature mapping as JSON.

    Example output:
    {
        "press_fibre_density": "operPressFibreDensity",
        "cooking_time": "operCookingTime"
    }
    """
    mes_feature_mapping: dict[str, str] = {}

    for _, row in mapping_df.iterrows():
        human_name = row.get("human_readable_name")
        technical_name = row.get("technical_name")

        if pd.isna(human_name) or pd.isna(technical_name):
            continue

        mes_name = normalize_mes_feature_name(
            str(human_name)
        )

        ml_name = str(technical_name).strip()

        if not mes_name or not ml_name:
            continue

        mes_feature_mapping[mes_name] = ml_name

    MES_FEATURE_MAPPING_FILE_PATH.write_text(
        json.dumps(
            mes_feature_mapping,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "MES feature mapping saved | path=%s | feature_count=%s",
        MES_FEATURE_MAPPING_FILE_PATH,
        len(mes_feature_mapping),
    )


def save_flattened_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Save parquet + CSV backup.
    """
    df_to_save = df.copy()

    object_cols = df_to_save.select_dtypes(
        include=["object"]
    ).columns

    for col in object_cols:
        df_to_save[col] = df_to_save[col].apply(
            lambda x: str(x).strip()
            if pd.notna(x)
            else x
        )

    df_to_save.to_parquet(
        FLATTENED_FILE_PATH,
        index=False,
    )

    csv_backup_path = (
        FLATTENED_FILE_PATH.with_suffix(".csv")
    )

    df_to_save.to_csv(
        csv_backup_path,
        index=False,
        encoding="utf-8-sig",
    )


@pipeline_step("Ingest raw KSoft files")
@log_dataframe_shape("Flattened dataframe")
def ingest_raw_ksoft_file() -> pd.DataFrame:
    ensure_directories()

    raw_df = load_raw_sheet()

    model_df = build_model_dataframe(raw_df)

    column_mapping_df = build_column_mapping(raw_df)

    save_column_mapping(column_mapping_df)

    save_mes_feature_mapping(column_mapping_df)

    save_flattened_dataframe(model_df)

    return model_df


if __name__ == "__main__":
    df = ingest_raw_ksoft_file()

    logger.info("Ingestion completed.")
    logger.info("Shape: %s", df.shape)
    logger.info(
        "First 20 columns: %s",
        df.columns[:20].tolist(),
    )