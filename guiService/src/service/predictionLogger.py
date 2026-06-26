from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from config.config import LOG_FILE


def _load_logs() -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []

    try:
        return json.loads(
            LOG_FILE.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        return []


def _write_logs(
    logs: list[dict[str, Any]],
) -> None:
    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_FILE.write_text(
        json.dumps(
            logs,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_prediction_log_entry(
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "source": data.get("source", "GUIService"),
        "mode": data.get("mode", "unknown"),

        "features": data.get("features"),
        "baseline": data.get("baseline"),
        "scenario": data.get("scenario"),

        "predictions": data.get("predictions"),
        "baseline_predictions": data.get("baseline_predictions"),
        "scenario_predictions": data.get("scenario_predictions"),

        "explanations": data.get("explanations"),
        "comparison": data.get("comparison"),
        "recommendations": data.get("recommendations"),
        "sensitivity_summary": data.get("sensitivity_summary"),
        "mini_trend_data": data.get("mini_trend_data"),

        "model_registry": data.get("model_registry"),
        "model_version": data.get("model_version"),
        "artifact_model_names": data.get("artifact_model_names"),

        "actual_values": data.get("actual_values"),
        "prediction_errors": data.get("prediction_errors"),
    }


def log_prediction(
    data: dict[str, Any],
) -> None:
    entry = build_prediction_log_entry(data)

    logs = _load_logs()
    logs.append(entry)

    _write_logs(logs)