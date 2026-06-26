from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import COMPARISON_REPORTS_DIR
from utils.decorators import pipeline_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def build_comparison_rows(
    results: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for result in results:
        rows.append(
            {
                "target_name": result.target_name,
                "algorithm_name": result.algorithm_name,
                "artifact_model_name": (
                    result.artifact_model_name
                ),
                "artifact_dir": result.artifact_dir,
                "validation_mae": (
                    result.metrics["validation"]["mae"]
                ),
                "validation_rmse": (
                    result.metrics["validation"]["rmse"]
                ),
                "validation_r2": (
                    result.metrics["validation"]["r2"]
                ),
                "test_mae": (
                    result.metrics["test"]["mae"]
                ),
                "test_rmse": (
                    result.metrics["test"]["rmse"]
                ),
                "test_r2": (
                    result.metrics["test"]["r2"]
                ),
                "train_rows": (
                    result.metrics["row_counts"]["train"]
                ),
                "val_rows": (
                    result.metrics["row_counts"]["val"]
                ),
                "test_rows": (
                    result.metrics["row_counts"]["test"]
                ),
                "split_type": result.metrics.get(
                    "split_type",
                    "",
                ),
            }
        )

    return rows


@pipeline_step("Save comparison summary")
def save_comparison_summary(
    results: list[Any],
    output_dir: str | Path = COMPARISON_REPORTS_DIR,
) -> None:
    rows = build_comparison_rows(results)

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df = pd.DataFrame(rows)

    csv_path = output_path / "model_comparison.csv"
    xlsx_path = output_path / "model_comparison.xlsx"
    json_path = output_path / "model_comparison.json"

    comparison_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_excel(
        xlsx_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Comparison summary saved | csv=%s | xlsx=%s | json=%s",
        csv_path,
        xlsx_path,
        json_path,
    )