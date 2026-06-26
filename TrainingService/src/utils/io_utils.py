from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    data: dict[str, Any],
    path: str | Path,
) -> None:
    path = Path(path)

    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def read_csv(
    path: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        **kwargs,
    )


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> None:
    path = Path(path)

    ensure_dir(path.parent)

    df.to_csv(
        path,
        index=False,
        **kwargs,
    )


def write_excel(
    df: pd.DataFrame,
    path: str | Path,
    **kwargs: Any,
) -> None:
    path = Path(path)

    ensure_dir(path.parent)

    df.to_excel(
        path,
        index=False,
        **kwargs,
    )


def safe_file_name(name: str) -> str:
    """
    Convert arbitrary text to a safer filename fragment.
    """
    cleaned = name.strip().replace(" ", "_")

    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        cleaned = cleaned.replace(char, "_")

    return cleaned


def write_dataframe_reports(
    df: pd.DataFrame,
    output_dir: str | Path,
    base_name: str,
) -> tuple[Path, Path]:
    output_path = ensure_dir(output_dir)

    csv_path = output_path / f"{base_name}.csv"
    xlsx_path = output_path / f"{base_name}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    return csv_path, xlsx_path