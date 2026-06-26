from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from core.model_registery import LoadedArtifact
from features.engineering import add_engineered_features
from service.input_validator import (
    analyze_input_features,
    build_ordered_feature_frame,
)
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def _extract_prediction_value(raw_prediction: Any) -> float:
    if hasattr(raw_prediction, "__len__"):
        return float(raw_prediction[0])

    return float(raw_prediction)


def _get_algorithm_name(artifact: LoadedArtifact) -> str:
    return (
        artifact.metadata.get("algorithm_name")
        or artifact.metadata.get("algorithm")
        or artifact.metadata.get("model_type")
        or "unknown"
    )


def _add_engineered_features_to_input(
    incoming_features: Dict[str, Any],
) -> Dict[str, Any]:
    input_df = pd.DataFrame([incoming_features])
    engineered_df = add_engineered_features(input_df)

    logger.info(
        "Feature engineering applied | input_count=%s | output_count=%s | engineered_keys=%s",
        len(incoming_features),
        len(engineered_df.columns),
        [
            col
            for col in engineered_df.columns
            if col not in incoming_features
        ],
    )

    return engineered_df.iloc[0].to_dict()

def _predict_from_artifact(
    target_name: str,
    incoming_features: Dict[str, Any],
    artifact: LoadedArtifact,
) -> Tuple[float, Dict[str, List[str]]]:

    enriched_features = _add_engineered_features_to_input(
        incoming_features
    )

    diagnostics = analyze_input_features(
        target_name=target_name,
        incoming_features=enriched_features,
        expected_features=artifact.features,
    )

    feature_frame = build_ordered_feature_frame(
        target_name=target_name,
        incoming_features=enriched_features,
        expected_features=artifact.features,
    )

    raw_prediction = artifact.predictor.predict(feature_frame)
    prediction_value = _extract_prediction_value(raw_prediction)

    return prediction_value, diagnostics


@service_step
def predict_single_model(
    target_name: str,
    incoming_features: Dict[str, Any],
    artifact: LoadedArtifact,
) -> Dict[str, Any]:
    prediction_value, diagnostics = _predict_from_artifact(
        target_name=target_name,
        incoming_features=incoming_features,
        artifact=artifact,
    )

    result = {
        "target": target_name,
        "model_name": artifact.model_name,
        "algorithm_name": _get_algorithm_name(artifact),
        "prediction": prediction_value,
        "used_feature_count": len(artifact.features),
        "missing_features": diagnostics["missing_features"],
        "extra_features": diagnostics["extra_features"],
        "invalid_numeric_features": diagnostics["invalid_numeric_features"],
        "null_features": diagnostics["null_features"],
        "forced_missing_features": diagnostics["forced_missing_features"],
        "serving_type": "single_model",
    }

    logger.info(
        "Single-model prediction completed | target=%s | model=%s | prediction=%s",
        target_name,
        artifact.model_name,
        prediction_value,
    )

    return result


@service_step
def predict_weighted_ensemble(
    target_name: str,
    incoming_features: Dict[str, Any],
    ensemble_members: List[Tuple[str, float, LoadedArtifact]],
) -> Dict[str, Any]:
    if not ensemble_members:
        raise ValueError(
            f"No ensemble members defined for target '{target_name}'."
        )

    prediction_sum = 0.0
    weight_sum = 0.0

    member_predictions: Dict[str, float] = {}
    member_models: Dict[str, str] = {}
    member_algorithms: Dict[str, str] = {}
    ensemble_weights: Dict[str, float] = {}

    missing_features: List[str] = []
    extra_features: List[str] = []
    invalid_numeric_features: List[str] = []
    null_features: List[str] = []
    forced_missing_features: List[str] = []

    max_used_feature_count = 0

    for member_key, weight, artifact in ensemble_members:
        if weight < 0:
            raise ValueError(
                f"Negative ensemble weight for member '{member_key}'."
            )

        prediction_value, diagnostics = _predict_from_artifact(
            target_name=target_name,
            incoming_features=incoming_features,
            artifact=artifact,
        )

        prediction_sum += weight * prediction_value
        weight_sum += weight

        member_predictions[member_key] = prediction_value
        member_models[member_key] = artifact.model_name
        member_algorithms[member_key] = _get_algorithm_name(artifact)
        ensemble_weights[member_key] = weight

        missing_features.extend(diagnostics["missing_features"])
        extra_features.extend(diagnostics["extra_features"])
        invalid_numeric_features.extend(
            diagnostics["invalid_numeric_features"]
        )
        null_features.extend(diagnostics["null_features"])
        forced_missing_features.extend(
            diagnostics["forced_missing_features"]
        )

        max_used_feature_count = max(
            max_used_feature_count,
            len(artifact.features),
        )

    if weight_sum <= 0:
        raise ValueError(
            f"Total ensemble weight must be greater than zero for target '{target_name}'."
        )

    prediction_value = prediction_sum / weight_sum

    result = {
        "target": target_name,
        "model_name": f"{target_name}_weighted_ensemble",
        "algorithm_name": "weighted_ensemble",
        "prediction": float(prediction_value),
        "used_feature_count": max_used_feature_count,
        "missing_features": sorted(set(missing_features)),
        "extra_features": sorted(set(extra_features)),
        "invalid_numeric_features": sorted(set(invalid_numeric_features)),
        "null_features": sorted(set(null_features)),
        "forced_missing_features": sorted(set(forced_missing_features)),
        "serving_type": "weighted_ensemble",
        "ensemble_members": member_models,
        "ensemble_algorithms": member_algorithms,
        "ensemble_weights": ensemble_weights,
        "member_predictions": member_predictions,
    }

    logger.info(
        "Weighted ensemble prediction completed | target=%s | members=%s | prediction=%s",
        target_name,
        list(member_models.keys()),
        prediction_value,
    )

    return result