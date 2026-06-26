from __future__ import annotations

from typing import Any

from config.config import (
    CONTROL_FEATURE_LABELS,
    CONTROL_FEATURE_UNITS,
    CORE_PROCESS_FEATURES,
    FORCED_PRODUCTION_FEATURES,
    PREDICTION_CLASS_HIGH,
    PREDICTION_CLASS_NORMAL,
    PREDICTION_CLASS_WARNING,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    ROW_CLASS_ALERT,
    ROW_CLASS_NORMAL,
    ROW_CLASS_WARNING,
    TARGET_LABELS,
    TARGET_THRESHOLDS,
    TARGET_UNITS,
)


def get_target_label(target_name: str) -> str:
    return TARGET_LABELS.get(target_name, target_name)


def get_target_unit(target_name: str) -> str:
    return TARGET_UNITS.get(target_name, "")


def build_prediction_status(
    target_name: str,
    prediction_value: float,
) -> tuple[str, str, str]:
    thresholds = TARGET_THRESHOLDS.get(target_name)

    if not thresholds:
        return ROW_CLASS_NORMAL, PREDICTION_CLASS_NORMAL, RISK_LOW

    danger_low = thresholds.get("danger_low")
    warning_low = thresholds.get("warning_low")
    warning_high = thresholds.get("warning_high")
    danger_high = thresholds.get("danger_high")

    if danger_low is not None and prediction_value <= danger_low:
        return ROW_CLASS_ALERT, PREDICTION_CLASS_HIGH, RISK_HIGH

    if danger_high is not None and prediction_value >= danger_high:
        return ROW_CLASS_ALERT, PREDICTION_CLASS_HIGH, RISK_HIGH

    if warning_low is not None and prediction_value <= warning_low:
        return ROW_CLASS_WARNING, PREDICTION_CLASS_WARNING, RISK_MEDIUM

    if warning_high is not None and prediction_value >= warning_high:
        return ROW_CLASS_WARNING, PREDICTION_CLASS_WARNING, RISK_MEDIUM

    return ROW_CLASS_NORMAL, PREDICTION_CLASS_NORMAL, RISK_LOW


def _extract_value(source: Any, feature_name: str) -> Any:
    if source is None:
        return None

    if isinstance(source, dict):
        return source.get(feature_name)

    return getattr(source, feature_name, None)


def build_control_features(
    source: Any,
    *,
    include_core_features: bool = True,
    include_forced_features: bool = True,
) -> dict[str, dict[str, Any]]:
    selected_features: list[str] = []

    if include_core_features:
        selected_features.extend(CORE_PROCESS_FEATURES)

    if include_forced_features:
        selected_features.extend(FORCED_PRODUCTION_FEATURES)

    selected_features = list(dict.fromkeys(selected_features))

    control_features: dict[str, dict[str, Any]] = {}

    for feature_name in selected_features:
        value = _extract_value(source, feature_name)

        if value is None:
            continue

        control_features[feature_name] = {
            "label": CONTROL_FEATURE_LABELS.get(feature_name, feature_name),
            "value": value,
            "unit": CONTROL_FEATURE_UNITS.get(feature_name, ""),
            "is_forced_production_feature": (
                feature_name in FORCED_PRODUCTION_FEATURES
            ),
        }

    return control_features


def build_prediction_display_value(
    prediction_value: float,
    target_name: str,
) -> str:
    unit = get_target_unit(target_name)

    formatted_value = f"{prediction_value:.2f}"

    if unit:
        return f"{formatted_value} {unit}"

    return formatted_value


def enrich_prediction_row(row: Any) -> Any:
    row_status_class, prediction_status_class, risk_level = build_prediction_status(
        target_name=row.target_name,
        prediction_value=float(row.prediction_value),
    )

    row.row_status_class = row_status_class
    row.prediction_status_class = prediction_status_class
    row.risk_level = risk_level

    row.target_label = get_target_label(row.target_name)
    row.target_unit = get_target_unit(row.target_name)

    row.prediction_display_value = build_prediction_display_value(
        prediction_value=float(row.prediction_value),
        target_name=row.target_name,
    )

    features_source = getattr(row, "features", None) or row

    row.control_features = build_control_features(
        features_source,
        include_core_features=True,
        include_forced_features=True,
    )

    return row


def enrich_prediction_rows(rows: list[Any]) -> list[Any]:
    return [
        enrich_prediction_row(row)
        for row in rows
    ]


def format_prediction_result_item(
    *,
    target_name: str,
    prediction_value: float,
    item: dict[str, Any],
) -> dict[str, Any]:
    row_class, prediction_class, risk_level = build_prediction_status(
        target_name=target_name,
        prediction_value=float(prediction_value),
    )

    return {
        **item,
        "target_name": target_name,
        "target_label": get_target_label(target_name),
        "target_unit": get_target_unit(target_name),
        "prediction": prediction_value,
        "prediction_display_value": build_prediction_display_value(
            prediction_value=float(prediction_value),
            target_name=target_name,
        ),
        "row_status_class": row_class,
        "prediction_status_class": prediction_class,
        "risk_level": risk_level,
    }