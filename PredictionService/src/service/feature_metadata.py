from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from core.model_registery import LoadedArtifact
from core.shared_model_registry import shared_model_registry
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)

model_registry = shared_model_registry


def _ensure_registry_loaded() -> None:
    if not model_registry.target_names():
        model_registry.load_all_from_registry()


def _detect_feature_column(row: Dict[str, Any]) -> str:
    for candidate in (
        "feature",
        "Feature",
        "feature_name",
        "FeatureName",
        "feature_names",
    ):
        if candidate in row:
            return candidate

    raise ValueError(
        f"Could not detect feature column. Found columns: {list(row.keys())}"
    )


def _detect_importance_column(row: Dict[str, Any]) -> str:
    for candidate in (
        "importance",
        "Importance",
        "score",
        "Score",
        "mean_abs_shap",
        "mean_abs_shap_value",
        "shap_importance",
    ):
        if candidate in row:
            return candidate

    raise ValueError(
        f"Could not detect importance column. Found columns: {list(row.keys())}"
    )


def _load_top_features_from_csv(
    csv_path: Path,
    top_n: int,
) -> List[str]:
    if not csv_path.exists():
        logger.warning(
            "Feature importance file not found: %s",
            csv_path,
        )
        return []

    rows: List[Dict[str, Any]] = []

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    if not rows:
        logger.warning(
            "Feature importance file is empty: %s",
            csv_path,
        )
        return []

    try:
        feature_column = _detect_feature_column(rows[0])
        importance_column = _detect_importance_column(rows[0])

    except ValueError as exc:
        logger.warning(
            "Could not parse feature importance CSV %s | %s",
            csv_path,
            exc,
        )
        return []

    def importance_value(row: Dict[str, Any]) -> float:
        try:
            return float(row.get(importance_column, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    rows_sorted = sorted(
        rows,
        key=importance_value,
        reverse=True,
    )

    return [
        row[feature_column]
        for row in rows_sorted[:top_n]
        if row.get(feature_column)
    ]


def _candidate_importance_files(
    artifact_dir: Path,
) -> List[Path]:
    return [
        artifact_dir / "shap_feature_importance.csv",
        artifact_dir / "feature_importance.csv",
        artifact_dir / "rf_feature_importance.csv",
        artifact_dir / "xgb_feature_importance.csv",
    ]


def _load_features_from_artifact_metadata(
    artifact: LoadedArtifact,
    top_n: int,
) -> List[str]:
    metadata_candidates = [
        "important_features",
        "top_features",
        "selected_features",
        "forced_production_features",
    ]

    for key in metadata_candidates:
        value = artifact.metadata.get(key)

        if isinstance(value, list) and value:
            return [
                str(feature)
                for feature in value[:top_n]
            ]

    return []


def _load_important_features_for_artifact(
    artifact: LoadedArtifact,
    top_n: int,
) -> List[str]:
    metadata_features = _load_features_from_artifact_metadata(
        artifact=artifact,
        top_n=top_n,
    )

    if metadata_features:
        return metadata_features

    for csv_path in _candidate_importance_files(
        artifact.artifact_dir
    ):
        features = _load_top_features_from_csv(
            csv_path=csv_path,
            top_n=top_n,
        )

        if features:
            return features

    return artifact.features[:top_n]


def _select_artifact_for_target(
    target_name: str,
) -> LoadedArtifact:
    entry = model_registry.get_registry_entry(target_name)

    if entry.serving_type == "single_model":
        return model_registry.get_artifact_for_target(target_name)

    if entry.serving_type == "weighted_ensemble":
        members = model_registry.get_ensemble_members_for_target(
            target_name
        )

        for member_key, _, artifact in members:
            if member_key.lower() == "xgb":
                return artifact

        return members[0][2]

    raise ValueError(
        f"Unsupported serving type '{entry.serving_type}' "
        f"for target '{target_name}'."
    )


@service_step
def load_important_features_by_target(
    top_n: int = 5,
) -> Dict[str, List[str]]:
    _ensure_registry_loaded()

    result: Dict[str, List[str]] = {}

    for target_name in model_registry.target_names():
        try:
            artifact = _select_artifact_for_target(
                target_name
            )

            result[target_name] = _load_important_features_for_artifact(
                artifact=artifact,
                top_n=top_n,
            )

        except Exception as exc:
            logger.exception(
                "Failed loading important features for target '%s': %s",
                target_name,
                exc,
            )

            result[target_name] = []

    return result