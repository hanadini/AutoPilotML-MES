from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.columns import TARGET_LABELS, REQUIRED_FINAL_EXPLAINABILITY_COLUMNS, NUMERIC_FINAL_EXPLAINABILITY_COLUMNS, \
    RECOMMENDED_ACTION_TEXT
from config.settings import (
    DATA_QUALITY_HIGH_MISSING_THRESHOLD,
    DATA_QUALITY_MODERATE_MISSING_THRESHOLD,
    FINAL_EXPLAINABILITY_DIR,
    FINAL_EXPLAINABILITY_FILE_NAME,
    PROCESS_RECOMMENDATION_DIR,
    PROCESS_RECOMMENDATION_TOP_N,
)
from utils.decorators import pipeline_step
from utils.io_utils import safe_file_name, write_dataframe_reports
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def load_final_explainability_report(
    path: str | Path = FINAL_EXPLAINABILITY_DIR / FINAL_EXPLAINABILITY_FILE_NAME,
) -> pd.DataFrame:
    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Final explainability report not found: {report_path}. "
            "Run save_final_explainability_report() first."
        )

    return pd.read_csv(report_path)


def _target_label(target: str) -> str:
    return TARGET_LABELS.get(target, target)


def _group_interpretation(
    process_group: str,
    target: str,
) -> str:
    target_label = _target_label(target)

    messages = {
        "fiber_preparation": (
            f"Fiber preparation appears to influence {target_label}. "
            "This may be related to cooking time, fiber temperature, "
            "fiber density, moisture, or steam/cooking conditions."
        ),
        "wood_mix": (
            f"Wood mix appears to influence {target_label}. "
            "This may reflect changes in leaf/conifer ratio or raw material composition."
        ),
        "mat_forming": (
            f"Mat forming appears to influence {target_label}. "
            "This may be related to mat weight, board density setup, "
            "catcher rate, or forming stability."
        ),
        "press_process": (
            f"Press process behavior appears to influence {target_label}. "
            "This may be related to press speed, press factor, lead position, "
            "or pressing stability."
        ),
        "glue_chemical": (
            f"Glue/chemical dosing appears to influence {target_label}. "
            "This may be related to glue, emulsion, hardener, or bonding behavior."
        ),
        "temperature_process": (
            f"Temperature process behavior appears to influence {target_label}. "
            "This may be related to press temperature, hot plate temperature, "
            "dryer temperature, or thermal stability."
        ),
        "speed_process": (
            f"Speed-related process behavior appears to influence {target_label}. "
            "This may be related to line speed, belt speed, discharge speed, "
            "or grinding speed."
        ),
    }

    return messages.get(
        process_group,
        (
            f"{process_group} appears to influence {target_label} "
            "and should be reviewed with production context."
        ),
    )


def _feature_comment(
    feature: str,
    direction: str | None,
) -> str:
    direction_messages = {
        "positive": (
            "Higher values are statistically associated with higher target values."
        ),
        "negative": (
            "Higher values are statistically associated with lower target values."
        ),
        "neutral": (
            "No clear positive or negative direction was detected."
        ),
        "unknown": (
            "Direction is not clearly available."
        ),
    }

    direction_text = direction_messages.get(
        str(direction),
        "Direction is not clearly available.",
    )

    return (
        f"Feature `{feature}` is one of the relevant signals. "
        f"{direction_text}"
    )


def _data_quality_comment(missing_ratio: Any) -> str:
    try:
        ratio = float(missing_ratio)
    except Exception:
        return "Data quality could not be evaluated."

    if ratio >= DATA_QUALITY_HIGH_MISSING_THRESHOLD:
        return (
            "High missing ratio. This signal should be treated carefully "
            "before operational use."
        )

    if ratio >= DATA_QUALITY_MODERATE_MISSING_THRESHOLD:
        return (
            "Moderate missing ratio. This signal is useful but should be "
            "monitored for data completeness."
        )

    return "Missing ratio is acceptable for analysis."


def _validate_input_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_FINAL_EXPLAINABILITY_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Final explainability report missing columns: {sorted(missing)}"
        )


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    for column in NUMERIC_FINAL_EXPLAINABILITY_COLUMNS:
        work_df[column] = pd.to_numeric(
            work_df[column],
            errors="coerce",
        )

    return work_df


@pipeline_step("Build process recommendation report")
def build_process_recommendation_report(
    df: pd.DataFrame,
    top_n_per_target: int = PROCESS_RECOMMENDATION_TOP_N,
) -> pd.DataFrame:
    _validate_input_schema(df)

    work_df = _coerce_numeric_columns(df)

    work_df = work_df.sort_values(
        by=["target", "final_explainability_score"],
        ascending=[True, False],
        na_position="last",
    )

    rows: list[dict[str, Any]] = []

    for target, target_df in work_df.groupby("target", dropna=False):
        top_df = target_df.head(top_n_per_target)

        for _, row in top_df.iterrows():
            process_group = row["process_group"]
            feature = row["feature"]
            direction = row.get("impact_direction")

            rows.append(
                {
                    "target": target,
                    "target_label": _target_label(target),
                    "algorithm": row.get("algorithm"),
                    "process_group": process_group,
                    "feature": feature,
                    "impact_direction": direction,
                    "correlation_with_target": row.get("correlation_with_target"),
                    "ml_importance": row.get("ml_importance"),
                    "mean_abs_shap": row.get("mean_abs_shap"),
                    "final_explainability_score": row.get(
                        "final_explainability_score"
                    ),
                    "missing_ratio": row.get("missing_ratio"),
                    "valid_pair_count": row.get("valid_pair_count"),
                    "process_interpretation": _group_interpretation(
                        process_group,
                        target,
                    ),
                    "feature_comment": _feature_comment(
                        feature,
                        direction,
                    ),
                    "data_quality_comment": _data_quality_comment(
                        row.get("missing_ratio")
                    ),
                    "recommended_action": RECOMMENDED_ACTION_TEXT,
                }
            )

    return pd.DataFrame(rows)


@pipeline_step("Save process recommendation report")
def save_process_recommendation_report(
    input_path: str | Path = (
        FINAL_EXPLAINABILITY_DIR / FINAL_EXPLAINABILITY_FILE_NAME
    ),
    output_dir: str | Path = PROCESS_RECOMMENDATION_DIR,
    top_n_per_target: int = PROCESS_RECOMMENDATION_TOP_N,
) -> pd.DataFrame:
    final_df = load_final_explainability_report(input_path)

    recommendation_df = build_process_recommendation_report(
        final_df,
        top_n_per_target=top_n_per_target,
    )

    write_dataframe_reports(
        df=recommendation_df,
        output_dir=output_dir,
        base_name="all_targets_process_recommendations",
    )

    for target in recommendation_df["target"].dropna().unique():
        target_df = recommendation_df[
            recommendation_df["target"] == target
        ].copy()

        write_dataframe_reports(
            df=target_df,
            output_dir=output_dir,
            base_name=safe_file_name(f"{target}_process_recommendations"),
        )

    logger.info(
        "Process recommendation reports saved | rows=%s | targets=%s",
        len(recommendation_df),
        recommendation_df["target"].nunique()
        if not recommendation_df.empty
        else 0,
    )

    return recommendation_df