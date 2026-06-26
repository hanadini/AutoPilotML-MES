from __future__ import annotations

import json
from typing import Any, Dict

import requests

from config.config import (
    DRIFT_STATUS_ENDPOINT,
    FEEDBACK_STATS_ENDPOINT,
    MODEL_PERFORMANCE_BY_TARGET_ENDPOINT,
    MODEL_PERFORMANCE_ENDPOINT,
    MONITORING_HEALTH_ENDPOINT,
    PREDICTION_STATS_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    RETRAINING_ADVISOR_ENDPOINT,
    SYSTEM_OVERVIEW_ENDPOINT,
    MONITORING_DASHBOARD_ENDPOINT,
    MONITORING_SNAPSHOTS_ENDPOINT,
)


class MonitoringServiceError(Exception):
    pass


NO_PROXY = {
    "http": None,
    "https": None,
}


def _get(
    *,
    url: str,
) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            proxies=NO_PROXY,
        )

    except requests.RequestException as exc:
        raise MonitoringServiceError(
            f"MonitoringService connection failed: {exc}"
        ) from exc

    if response.status_code != 200:
        raise MonitoringServiceError(
            f"MonitoringService returned "
            f"status={response.status_code} "
            f"body={response.text}"
        )

    try:
        return response.json()

    except json.JSONDecodeError as exc:
        raise MonitoringServiceError(
            "MonitoringService returned invalid JSON."
        ) from exc


def monitoring_health_check() -> Dict[str, Any]:
    return _get(
        url=MONITORING_HEALTH_ENDPOINT,
    )


def get_system_overview() -> Dict[str, Any]:
    return _get(
        url=SYSTEM_OVERVIEW_ENDPOINT,
    )


def get_prediction_stats() -> Dict[str, Any]:
    return _get(
        url=PREDICTION_STATS_ENDPOINT,
    )


def get_feedback_stats() -> Dict[str, Any]:
    return _get(
        url=FEEDBACK_STATS_ENDPOINT,
    )


def get_model_performance() -> Dict[str, Any]:
    return _get(
        url=MODEL_PERFORMANCE_ENDPOINT,
    )


def get_model_performance_by_target() -> Dict[str, Any]:
    return _get(
        url=MODEL_PERFORMANCE_BY_TARGET_ENDPOINT,
    )


def get_retraining_advisor() -> Dict[str, Any]:
    return _get(
        url=RETRAINING_ADVISOR_ENDPOINT,
    )

def get_drift_status() -> Dict[str, Any]:
    return _get(url=DRIFT_STATUS_ENDPOINT)


def get_monitoring_dashboard() -> Dict[str, Any]:
    return _get(
        url=MONITORING_DASHBOARD_ENDPOINT,
    )

def get_monitoring_snapshots() -> Dict[str, Any]:
    return _get(
        url=MONITORING_SNAPSHOTS_ENDPOINT,
    )

def _post(
    *,
    url: str,
) -> Dict[str, Any]:
    try:
        response = requests.post(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            proxies=NO_PROXY,
        )

    except requests.RequestException as exc:
        raise MonitoringServiceError(
            f"MonitoringService connection failed: {exc}"
        ) from exc

    if response.status_code != 200:
        raise MonitoringServiceError(
            f"MonitoringService returned "
            f"status={response.status_code} "
            f"body={response.text}"
        )

    try:
        return response.json()

    except json.JSONDecodeError as exc:
        raise MonitoringServiceError(
            "MonitoringService returned invalid JSON."
        ) from exc


def create_monitoring_snapshot() -> Dict[str, Any]:
    return _post(
        url=MONITORING_SNAPSHOTS_ENDPOINT,
    )
