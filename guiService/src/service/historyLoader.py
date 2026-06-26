from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from config.config import (
    ALL_TARGETS,
    HISTORY_DEFAULT_LIMIT,
    LOG_FILE,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    TARGET_LABELS, ALIASES,
)


def _format_timestamp(timestamp_raw: str | None) -> str:
    if not timestamp_raw:
        return "-"

    try:
        parsed_timestamp = datetime.fromisoformat(timestamp_raw)
        return parsed_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_raw


def _load_logs() -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    try:
        return json.loads(
            LOG_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return []


def _normalize_risk_level(risk: str | None) -> str | None:
    if not risk:
        return None

    risk_value = risk.upper().strip()

    aliases = ALIASES

    return aliases.get(risk_value, risk_value)


def _target_label(target_name: str) -> str:
    return TARGET_LABELS.get(target_name, target_name)


def _filter_recommendations(
    recommendations: Any,
    *,
    risk: str | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    if not isinstance(recommendations, dict):
        return {}

    normalized_risk = _normalize_risk_level(risk)

    filtered: dict[str, Any] = {}

    for target_name, recommendation in recommendations.items():
        if not isinstance(recommendation, dict):
            continue

        if target and target_name != target:
            continue

        recommendation_risk = _normalize_risk_level(
            recommendation.get("risk_level")
        )

        if normalized_risk and recommendation_risk != normalized_risk:
            continue

        enriched_recommendation = dict(recommendation)
        enriched_recommendation["risk_level"] = recommendation_risk
        enriched_recommendation["target_label"] = _target_label(target_name)

        filtered[target_name] = enriched_recommendation

    return filtered


def _enrich_history_item(
    item: dict[str, Any],
    *,
    risk: str | None = None,
    target: str | None = None,
) -> dict[str, Any] | None:
    recommendations = _filter_recommendations(
        item.get("recommendations"),
        risk=risk,
        target=target,
    )

    if risk or target:
        if not recommendations:
            return None

    enriched_item = dict(item)
    enriched_item["recommendations"] = recommendations
    enriched_item["timestamp_formatted"] = _format_timestamp(
        item.get("timestamp")
    )

    return enriched_item


def load_history(
    limit: int = HISTORY_DEFAULT_LIMIT,
    risk: str | None = None,
    target: str | None = None,
) -> list[dict[str, Any]]:
    logs = _load_logs()

    if not logs:
        return []

    filtered_logs: list[dict[str, Any]] = []

    for item in reversed(logs):
        enriched_item = _enrich_history_item(
            item,
            risk=risk,
            target=target,
        )

        if enriched_item is None:
            continue

        filtered_logs.append(enriched_item)

        if len(filtered_logs) >= limit:
            break

    return filtered_logs


def load_history_entry_by_timestamp(
    timestamp: str,
) -> dict[str, Any] | None:
    logs = _load_logs()

    for item in logs:
        if item.get("timestamp") == timestamp:
            enriched_item = dict(item)
            enriched_item["timestamp_formatted"] = _format_timestamp(
                item.get("timestamp")
            )
            return enriched_item

    return None


def get_history_target_options() -> list[dict[str, str]]:
    return [
        {
            "name": target,
            "label": TARGET_LABELS.get(target, target),
        }
        for target in ALL_TARGETS
    ]