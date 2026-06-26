from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.model_registery import LoadedArtifact
from core.shared_model_registry import shared_model_registry
from service.explainer import explain_single_model_prediction
from service.predictor import predict_single_model, predict_weighted_ensemble
from service.recommendation_engine import build_shap_recommendations
from service.risk_level import calculate_risk
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)



model_registry = shared_model_registry


def _ensure_registry_loaded() -> None:
    if not model_registry.target_names():
        model_registry.load_all_from_registry()


def _select_explanation_artifact(
    ensemble_members: List[Tuple[str, float, LoadedArtifact]],
) -> LoadedArtifact:
    for member_key, _, artifact in ensemble_members:
        if member_key.lower() == "xgb":
            return artifact

    return ensemble_members[0][2]


def _get_explanation_algorithm(
    artifact: LoadedArtifact,
) -> str:
    return (
        artifact.metadata.get("algorithm_name")
        or artifact.metadata.get("algorithm")
        or artifact.metadata.get("model_type")
        or "unknown"
    )


@service_step
def explain_prediction(
    *,
    target: str,
    incoming_features: Dict[str, Any],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Prediction + SHAP explanation + recommendation layer.

    This function must never let SHAP failure block prediction serving.
    If prediction works but SHAP fails, it returns prediction with an empty
    explanation and a diagnostic recommendation.
    """

    logger.info(
        "Explain prediction started | target=%s",
        target,
    )

    _ensure_registry_loaded()

    registry_entry = model_registry.get_registry_entry(target)

    if registry_entry.serving_type == "single_model":
        artifact = model_registry.get_artifact_for_target(target)

        prediction_result = predict_single_model(
            target_name=target,
            incoming_features=incoming_features,
            artifact=artifact,
        )

        explanation_artifact = artifact

    elif registry_entry.serving_type == "weighted_ensemble":
        ensemble_members = model_registry.get_ensemble_members_for_target(
            target
        )

        prediction_result = predict_weighted_ensemble(
            target_name=target,
            incoming_features=incoming_features,
            ensemble_members=ensemble_members,
        )

        explanation_artifact = _select_explanation_artifact(
            ensemble_members
        )

    else:
        raise ValueError(
            f"Unsupported serving type for target '{target}': "
            f"{registry_entry.serving_type}"
        )

    prediction_value = prediction_result["prediction"]

    risk_level = calculate_risk(
        target,
        prediction_value,
    )

    explanation: List[Dict[str, Any]] = []
    recommendations: List[Dict[str, Any]] = []

    try:
        explanation = explain_single_model_prediction(
            target_name=target,
            artifact=explanation_artifact,
            incoming_features=incoming_features,
            top_n=top_n,
        )

        recommendations = build_shap_recommendations(
            target=target,
            risk_level=risk_level,
            explanation=explanation,
            top_n=3,
        )

    except Exception as exc:
        logger.exception(
            "SHAP/recommendation layer failed | target=%s | "
            "explanation_model=%s | error=%s",
            target,
            explanation_artifact.model_name,
            str(exc),
        )

        recommendations = [
            {
                "feature": None,
                "feature_label": "Explainability layer",
                "impact": 0.0,
                "direction": "unknown",
                "risk_level": risk_level,
                "message": (
                    "Prediction was generated successfully, but SHAP "
                    f"explanation failed for target '{target}'. "
                    f"Technical detail: {str(exc)}"
                ),
            }
        ]

    result: Dict[str, Any] = {
        "target": target,
        "prediction": prediction_value,
        "risk_level": risk_level,
        "model_name": prediction_result["model_name"],
        "algorithm_name": prediction_result["algorithm_name"],
        "serving_type": prediction_result.get("serving_type"),
        "used_feature_count": prediction_result["used_feature_count"],
        "missing_features": prediction_result.get("missing_features", []),
        "extra_features": prediction_result.get("extra_features", []),
        "invalid_numeric_features": prediction_result.get(
            "invalid_numeric_features",
            [],
        ),
        "null_features": prediction_result.get("null_features", []),
        "forced_missing_features": prediction_result.get(
            "forced_missing_features",
            [],
        ),
        "explanation_model_name": explanation_artifact.model_name,
        "explanation_algorithm": _get_explanation_algorithm(
            explanation_artifact
        ),
        "explanation": explanation,
        "recommendations": recommendations,
    }

    if "ensemble_members" in prediction_result:
        result["ensemble_members"] = prediction_result["ensemble_members"]
        result["ensemble_weights"] = prediction_result["ensemble_weights"]
        result["member_predictions"] = prediction_result["member_predictions"]

    logger.info(
        "Explain prediction completed | target=%s | model=%s | "
        "risk=%s | explanation_items=%s",
        target,
        result["model_name"],
        risk_level,
        len(explanation),
    )

    return result