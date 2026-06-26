from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

from config.settings import (
    ALLOW_EXTRA_FEATURES,
    ENABLE_FORCED_PRODUCTION_FEATURE_CHECK,
    ENABLE_NAN_GUARD,
    ENABLE_TYPE_COERCION,
    FORCED_PRODUCTION_FEATURES_BY_TARGET,
    STRICT_FEATURE_ORDER,
)
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def analyze_input_features(
    target_name: str,
    incoming_features: Dict[str, Any],
    expected_features: List[str],
) -> Dict[str, List[str]]:
    incoming_keys = set(incoming_features.keys())
    expected_keys = set(expected_features)

    missing_features = sorted(expected_keys - incoming_keys)
    extra_features = sorted(incoming_keys - expected_keys)

    invalid_numeric_features: List[str] = []
    null_features: List[str] = []

    for key in expected_features:
        if key not in incoming_features:
            continue

        value = incoming_features.get(key)

        if value is None:
            null_features.append(key)
            continue

        if isinstance(value, bool):
            invalid_numeric_features.append(key)
            continue

        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                invalid_numeric_features.append(key)
            continue

        try:
            converted = float(value)

            if not math.isfinite(converted):
                invalid_numeric_features.append(key)

        except (TypeError, ValueError):
            invalid_numeric_features.append(key)

    forced_features = FORCED_PRODUCTION_FEATURES_BY_TARGET.get(
        target_name,
        [],
    )

    forced_missing_features = [
        feature
        for feature in forced_features
        if feature in expected_keys and feature not in incoming_keys
    ]

    return {
        "missing_features": missing_features,
        "extra_features": extra_features,
        "invalid_numeric_features": sorted(invalid_numeric_features),
        "null_features": sorted(null_features),
        "forced_missing_features": sorted(forced_missing_features),
    }


@service_step
def validate_input_features(
    target_name: str,
    incoming_features: Dict[str, Any],
    expected_features: List[str],
) -> Dict[str, List[str]]:
    diagnostics = analyze_input_features(
        target_name=target_name,
        incoming_features=incoming_features,
        expected_features=expected_features,
    )

    if diagnostics["missing_features"]:
        logger.warning(
            "Missing model feature(s) will be filled with NaN | "
            "target=%s | missing_count=%s | missing=%s",
            target_name,
            len(diagnostics["missing_features"]),
            diagnostics["missing_features"],
        )

    if diagnostics["extra_features"] and not ALLOW_EXTRA_FEATURES:
        raise ValueError(
            f"Unexpected extra feature(s): {diagnostics['extra_features']}"
        )

    if diagnostics["invalid_numeric_features"]:
        raise ValueError(
            "Invalid numeric feature(s): "
            f"{diagnostics['invalid_numeric_features']}"
        )

    if ENABLE_NAN_GUARD and diagnostics["null_features"]:
        raise ValueError(
            "Null/NaN feature(s) are not allowed for provided input values: "
            f"{diagnostics['null_features']}"
        )

    if (
        ENABLE_FORCED_PRODUCTION_FEATURE_CHECK
        and diagnostics["forced_missing_features"]
    ):
        raise ValueError(
            "Missing forced production feature(s): "
            f"{diagnostics['forced_missing_features']}"
        )

    return diagnostics


@service_step
def build_ordered_feature_frame(
    target_name: str,
    incoming_features: Dict[str, Any],
    expected_features: List[str],
) -> pd.DataFrame:
    diagnostics = validate_input_features(
        target_name=target_name,
        incoming_features=incoming_features,
        expected_features=expected_features,
    )

    ordered_row: Dict[str, Any] = {}

    for feature_name in expected_features:
        if feature_name not in incoming_features:
            ordered_row[feature_name] = float("nan")
            continue

        value = incoming_features.get(feature_name)

        if value is None:
            ordered_row[feature_name] = float("nan")
            continue

        if ENABLE_TYPE_COERCION:
            ordered_row[feature_name] = float(value)
        else:
            ordered_row[feature_name] = value

    frame = pd.DataFrame(
        [ordered_row],
        columns=expected_features,
    )

    if STRICT_FEATURE_ORDER:
        frame = frame[expected_features]

    logger.info(
        "Ordered feature frame built | target=%s | feature_count=%s | "
        "missing_filled=%s",
        target_name,
        len(expected_features),
        len(diagnostics["missing_features"]),
    )

    return frame