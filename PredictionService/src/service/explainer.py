from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from config.settings import WATCH_FEATURES
from core.model_registery import LoadedArtifact
from features.engineering import add_engineered_features
from service.input_validator import build_ordered_feature_frame
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def _extract_pipeline_components(
    artifact: LoadedArtifact,
):
    pipeline = artifact.predictor

    if not hasattr(pipeline, "named_steps"):
        raise ValueError(
            f"Artifact '{artifact.model_name}' is not a sklearn Pipeline."
        )

    imputer = pipeline.named_steps.get("imputer")
    model = pipeline.named_steps.get("model")

    if imputer is None:
        raise ValueError(
            f"Pipeline missing 'imputer' step for "
            f"artifact '{artifact.model_name}'."
        )

    if model is None:
        raise ValueError(
            f"Pipeline missing 'model' step for "
            f"artifact '{artifact.model_name}'."
        )

    return imputer, model


def _add_engineered_features_to_input(
    incoming_features: Dict[str, Any],
) -> Dict[str, Any]:
    input_df = pd.DataFrame([incoming_features])

    engineered_df = add_engineered_features(
        input_df
    )

    enriched_features = engineered_df.iloc[0].to_dict()

    engineered_keys = [
        key
        for key in enriched_features.keys()
        if key not in incoming_features
    ]

    logger.info(
        "Feature engineering applied for SHAP | input_count=%s | output_count=%s | engineered_count=%s | engineered_keys=%s",
        len(incoming_features),
        len(enriched_features),
        len(engineered_keys),
        engineered_keys,
    )

    logger.info(
        "Engineered feature check for SHAP | pressPressureC_mean=%s | pressPressureMid_mean=%s | pressPressureGlobal_mean=%s | tempGlobal_max=%s | thicknessClosed_mean=%s",
        enriched_features.get("pressPressureC_mean"),
        enriched_features.get("pressPressureMid_mean"),
        enriched_features.get("pressPressureGlobal_mean"),
        enriched_features.get("tempGlobal_max"),
        enriched_features.get("thicknessClosed_mean"),
    )

    return enriched_features


@service_step
def explain_single_model_prediction(
    *,
    target_name: str,
    artifact: LoadedArtifact,
    incoming_features: Dict[str, Any],
    top_n: int = 5,
) -> List[Dict[str, Any]]:

    logger.info(
        "EXPLAINER_ENTERED | target=%s",
        target_name,
    )

    try:
        import shap

    except ImportError:
        logger.warning(
            "SHAP is not installed. Run: pip install shap"
        )
        return []

    try:
        imputer, model = _extract_pipeline_components(
            artifact
        )

        enriched_features = _add_engineered_features_to_input(
            incoming_features
        )

        print(
            "DEBUG_ENGINEERED:",
            enriched_features.get("pressPressureC_mean"),
            enriched_features.get("pressPressureMid_mean"),
            enriched_features.get("pressPressureGlobal_mean"),
            enriched_features.get("tempGlobal_max"),
            enriched_features.get("thicknessClosed_mean"),
        )

        input_frame = build_ordered_feature_frame(
            target_name=target_name,
            incoming_features=enriched_features,
            expected_features=artifact.features,
        )

        logger.info(
            "SHAP input frame feature check | target=%s | pressPressureC_mean=%s | pressPressureMid_mean=%s | pressPressureGlobal_mean=%s | tempGlobal_max=%s | thicknessClosed_mean=%s",
            target_name,
            (
                input_frame["pressPressureC_mean"].iloc[0]
                if "pressPressureC_mean" in input_frame.columns
                else None
            ),
            (
                input_frame["pressPressureMid_mean"].iloc[0]
                if "pressPressureMid_mean" in input_frame.columns
                else None
            ),
            (
                input_frame["pressPressureGlobal_mean"].iloc[0]
                if "pressPressureGlobal_mean" in input_frame.columns
                else None
            ),
            (
                input_frame["tempGlobal_max"].iloc[0]
                if "tempGlobal_max" in input_frame.columns
                else None
            ),
            (
                input_frame["thicknessClosed_mean"].iloc[0]
                if "thicknessClosed_mean" in input_frame.columns
                else None
            ),
        )

        transformed_input = imputer.transform(
            input_frame
        )

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            transformed_input
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        impacts = shap_values[0]

        explanation: List[Dict[str, Any]] = []

        for feature_name, feature_value, impact in zip(
            artifact.features,
            input_frame.iloc[0].tolist(),
            impacts,
        ):
            explanation.append(
                {
                    "feature": feature_name,
                    "value": (
                        None
                        if pd.isna(feature_value)
                        else float(feature_value)
                        if isinstance(feature_value, (int, float))
                        else feature_value
                    ),
                    "impact": float(impact),
                    "absolute_impact": abs(float(impact)),
                    "direction": (
                        "increase"
                        if impact >= 0
                        else "decrease"
                    ),
                    "is_watched_feature": (
                        feature_name in WATCH_FEATURES
                    ),
                }
            )

        explanation.sort(
            key=lambda item: item["absolute_impact"],
            reverse=True,
        )

        logger.info(
            "SHAP explanation generated | target=%s | model=%s | feature_count=%s | top_n=%s",
            target_name,
            artifact.model_name,
            len(artifact.features),
            top_n,
        )

        top_items = explanation[:top_n]

        watched_items = [
            item
            for item in explanation
            if item["feature"] in WATCH_FEATURES
            and item not in top_items
        ]

        return top_items + watched_items

    except Exception as exc:
        logger.exception(
            "SHAP explanation failed | target=%s | model=%s | error=%s",
            target_name,
            artifact.model_name,
            str(exc),
        )

        return []