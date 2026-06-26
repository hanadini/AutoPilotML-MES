from __future__ import annotations

from typing import Any, Dict, List, Set

from config.settings import FEATURE_LABELS, WATCH_FEATURES
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def _human_feature_name(feature: str) -> str:
    for key, label in FEATURE_LABELS.items():
        if key.lower() in feature.lower():
            return label

    return feature


def _build_message(
    *,
    target: str,
    human_feature: str,
    impact: float,
) -> str:
    if target == "labDensityAverage":
        if impact > 0:
            return (
                f"{human_feature} is increasing predicted density. "
                "Review whether it can be reduced safely."
            )

        return (
            f"{human_feature} is reducing predicted density. "
            "Review whether it should be stabilized."
        )

    if target == "labBendingAvg":
        if impact > 0:
            return (
                f"{human_feature} is supporting predicted bending strength."
            )

        return (
            f"{human_feature} is reducing predicted bending strength. "
            "Check process stability."
        )

    if target == "labEModulAvg":
        if impact > 0:
            return (
                f"{human_feature} is supporting predicted E-modulus."
            )

        return (
            f"{human_feature} is reducing predicted E-modulus. "
            "Check process stability."
        )

    if target == "labTensileAvg":
        if impact > 0:
            return (
                f"{human_feature} is supporting predicted tensile strength."
            )

        return (
            f"{human_feature} is reducing predicted tensile strength. "
            "Check process stability."
        )

    if target == "labSurfaceSoundnessAvg":
        if impact > 0:
            return (
                f"{human_feature} is supporting predicted surface soundness."
            )

        return (
            f"{human_feature} is reducing predicted surface soundness. "
            "Check surface-related process stability."
        )

    direction = "increases" if impact > 0 else "decreases"

    return f"{human_feature} {direction} the predicted value."


@service_step
def build_shap_recommendations(
    *,
    target: str,
    risk_level: str,
    explanation: List[Dict[str, Any]],
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []
    selected_features: Set[str] = set()

    selected_items: List[Dict[str, Any]] = list(
        explanation[:top_n]
    )

    for item in explanation:
        feature = str(item.get("feature", ""))

        if (
            feature in WATCH_FEATURES
            and feature not in selected_features
            and item not in selected_items
        ):
            selected_items.append(item)

    selected_items_sorted = sorted(
        selected_items,
        key=lambda item: abs(
            float(item.get("impact", 0.0))
        ),
        reverse=True,
    )

    for item in selected_items_sorted:
        feature = str(item.get("feature", ""))

        if not feature or feature in selected_features:
            continue

        impact = float(item.get("impact", 0.0))
        human_feature = _human_feature_name(feature)
        direction = "increases" if impact > 0 else "decreases"

        recommendations.append(
            {
                "feature": feature,
                "feature_label": human_feature,
                "impact": impact,
                "direction": direction,
                "risk_level": risk_level,
                "message": _build_message(
                    target=target,
                    human_feature=human_feature,
                    impact=impact,
                ),
                "is_watched_feature": feature in WATCH_FEATURES,
            }
        )

        selected_features.add(feature)

    logger.info(
        "Built SHAP recommendations | target=%s | risk=%s | count=%s",
        target,
        risk_level,
        len(recommendations),
    )

    return recommendations