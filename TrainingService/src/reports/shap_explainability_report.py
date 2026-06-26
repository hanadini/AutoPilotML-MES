from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    ARTIFACTS_DIR,
    SHAP_EXPLAINABILITY_DIR,
    SHAP_MAX_ROWS,
)
from utils.decorators import pipeline_step
from utils.io_utils import safe_file_name, write_dataframe_reports
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_final_model(model: Any) -> Any:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]

    return model


def _prepare_model_input(
    model: Any,
    X: pd.DataFrame,
):
    if hasattr(model, "named_steps") and "imputer" in model.named_steps:
        imputer = model.named_steps["imputer"]
        return imputer.transform(X)

    return X


def _extract_feature_names(features_data: dict[str, Any] | list[Any]) -> list[str]:
    if isinstance(features_data, dict):
        return features_data.get("feature_names", [])

    if isinstance(features_data, list):
        return features_data

    raise ValueError(
        f"Unsupported features.json format: {type(features_data)}"
    )


def _safe_shap_importance(
    *,
    model: Any,
    X: pd.DataFrame,
    max_rows: int = SHAP_MAX_ROWS,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it first with: pip install shap"
        ) from exc

    if len(X) > max_rows:
        X_sample = X.sample(
            n=max_rows,
            random_state=42,
        )
    else:
        X_sample = X.copy()

    final_model = _get_final_model(model)
    X_for_shap = _prepare_model_input(model, X_sample)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_for_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame(
        {
            "feature": X_sample.columns.tolist(),
            "mean_abs_shap": mean_abs_shap,
        }
    )

    shap_df = (
        shap_df
        .sort_values(
            by="mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    shap_df["shap_rank"] = range(1, len(shap_df) + 1)

    return shap_df


@pipeline_step("Build SHAP report for artifact")
def build_shap_report_for_artifact(
    *,
    artifact_dir: str | Path,
    df: pd.DataFrame,
    max_rows: int = SHAP_MAX_ROWS,
) -> pd.DataFrame:
    artifact_path = Path(artifact_dir)

    model_path = artifact_path / "model.joblib"
    features_path = artifact_path / "features.json"
    metadata_path = artifact_path / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)

    features_data = _load_json(features_path)
    metadata = _load_json(metadata_path)

    feature_names = _extract_feature_names(features_data)

    if not feature_names:
        raise ValueError(
            f"No feature names found in features.json for {artifact_path.name}"
        )

    existing_features = [
        feature
        for feature in feature_names
        if feature in df.columns
    ]

    if not existing_features:
        raise ValueError(
            f"No artifact features found in dataframe for {artifact_path.name}"
        )

    X = df[existing_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    shap_df = _safe_shap_importance(
        model=model,
        X=X,
        max_rows=max_rows,
    )

    shap_df["target"] = metadata.get("target_name")
    shap_df["algorithm"] = metadata.get("algorithm")
    shap_df["artifact_model_name"] = metadata.get(
        "model_name",
        artifact_path.name,
    )
    shap_df["artifact_dir"] = str(artifact_path)

    ordered_columns = [
        "target",
        "algorithm",
        "artifact_model_name",
        "feature",
        "shap_rank",
        "mean_abs_shap",
        "artifact_dir",
    ]

    return shap_df[ordered_columns]


def find_artifact_dirs(
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> list[Path]:
    artifacts_path = Path(artifacts_dir)

    if not artifacts_path.exists():
        return []

    artifact_dirs: list[Path] = []

    for path in artifacts_path.iterdir():
        if not path.is_dir():
            continue

        if (
            (path / "model.joblib").exists()
            and (path / "features.json").exists()
        ):
            artifact_dirs.append(path)

    return artifact_dirs


@pipeline_step("Save SHAP explainability reports")
def save_shap_explainability_reports(
    *,
    df: pd.DataFrame,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
    output_dir: str | Path = SHAP_EXPLAINABILITY_DIR,
    max_rows: int = SHAP_MAX_ROWS,
) -> pd.DataFrame:
    shap_reports: list[pd.DataFrame] = []

    for artifact_dir in find_artifact_dirs(artifacts_dir):
        try:
            shap_df = build_shap_report_for_artifact(
                artifact_dir=artifact_dir,
                df=df,
                max_rows=max_rows,
            )
        except Exception as exc:
            logger.warning(
                "SHAP skipped | artifact=%s | error=%s",
                artifact_dir.name,
                exc,
            )
            continue

        target = shap_df["target"].iloc[0]
        algorithm = shap_df["algorithm"].iloc[0]
        artifact_name = shap_df["artifact_model_name"].iloc[0]

        base_name = safe_file_name(
            f"{target}_{algorithm}_{artifact_name}_shap_importance"
        )

        write_dataframe_reports(
            df=shap_df,
            output_dir=output_dir,
            base_name=base_name,
        )

        shap_reports.append(shap_df)

    if shap_reports:
        combined_df = (
            pd.concat(shap_reports, axis=0)
            .reset_index(drop=True)
        )
    else:
        combined_df = pd.DataFrame()

    if not combined_df.empty:
        write_dataframe_reports(
            df=combined_df,
            output_dir=output_dir,
            base_name="all_targets_shap_importance",
        )

    logger.info(
        "SHAP explainability reports completed | artifacts=%s | rows=%s",
        len(shap_reports),
        len(combined_df),
    )

    return combined_df