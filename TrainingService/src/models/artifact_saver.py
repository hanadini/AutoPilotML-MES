from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils.decorators import pipeline_step


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def make_model_artifact_dir(
    artifacts_root: str | Path,
    model_name: str,
) -> Path:
    root = ensure_dir(artifacts_root)

    model_dir = root / model_name

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return model_dir


def save_model(
    model: Any,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_dir(output_dir)

    model_path = output_dir / "model.joblib"

    joblib.dump(model, model_path)

    return model_path


def save_preprocessor(
    preprocessor: Any,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_dir(output_dir)

    preprocessor_path = (
        output_dir / "preprocessor.joblib"
    )

    joblib.dump(preprocessor, preprocessor_path)

    return preprocessor_path


def save_features(
    feature_names: list[str],
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_dir(output_dir)

    features_path = output_dir / "features.json"

    features_data = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
    }

    with open(
        features_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            features_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return features_path


def save_metrics(
    metrics: dict,
    output_dir: str | Path,
) -> Path:
    output_dir = ensure_dir(output_dir)

    metrics_path = output_dir / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metrics_path


def save_metadata(
    *,
    output_dir: str | Path,
    model_name: str,
    target_name: str,
    algorithm: str,
    feature_names: list[str],
    notes: str | None = None,
    extra: dict | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)

    metadata_path = output_dir / "metadata.json"

    metadata = {
        "model_name": model_name,
        "target_name": target_name,
        "algorithm": algorithm,
        "feature_count": len(feature_names),
        "training_timestamp": (
            datetime.utcnow().isoformat() + "Z"
        ),
        "notes": notes or "",
    }

    if extra:
        metadata.update(extra)

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path


def save_feature_importance(
    feature_importance_df: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = ensure_dir(output_dir)

    csv_path = (
        output_dir / "feature_importance.csv"
    )

    xlsx_path = (
        output_dir / "feature_importance.xlsx"
    )

    feature_importance_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    feature_importance_df.to_excel(
        xlsx_path,
        index=False,
    )

    return csv_path, xlsx_path


@pipeline_step("Save full artifact bundle")
def save_full_artifact_bundle(
    *,
    artifacts_root: str | Path,
    model_name: str,
    model: Any,
    feature_names: list[str],
    metrics: dict,
    target_name: str,
    algorithm: str,
    preprocessor: Any | None = None,
    feature_importance_df: pd.DataFrame | None = None,
    notes: str | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """
    Save a full production-ready
    model artifact bundle.
    """
    model_dir = make_model_artifact_dir(
        artifacts_root,
        model_name,
    )

    save_model(model, model_dir)

    save_features(
        feature_names,
        model_dir,
    )

    save_metrics(
        metrics,
        model_dir,
    )

    save_metadata(
        output_dir=model_dir,
        model_name=model_name,
        target_name=target_name,
        algorithm=algorithm,
        feature_names=feature_names,
        notes=notes,
        extra=extra_metadata,
    )

    if preprocessor is not None:
        save_preprocessor(
            preprocessor,
            model_dir,
        )

    if (
        feature_importance_df is not None
        and not feature_importance_df.empty
    ):
        save_feature_importance(
            feature_importance_df,
            model_dir,
        )

    return model_dir