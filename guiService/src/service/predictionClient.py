from __future__ import annotations

import json
from typing import Any, Dict

import requests

from config.config import (
    EXPLAIN_ENDPOINT,
    HEALTH_ENDPOINT,
    MODELS_ENDPOINT,
    PREDICT_ALL_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
)


class PredictionServiceError(Exception):
    pass


NO_PROXY = {
    "http": None,
    "https": None,
}


def _post(
    *,
    url: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
            proxies=NO_PROXY,
        )

    except requests.RequestException as exc:
        raise PredictionServiceError(
            f"PredictionService connection failed: {exc}"
        ) from exc

    if response.status_code != 200:
        raise PredictionServiceError(
            f"PredictionService returned "
            f"status={response.status_code} "
            f"body={response.text}"
        )

    try:
        return response.json()

    except json.JSONDecodeError as exc:
        raise PredictionServiceError(
            "PredictionService returned invalid JSON."
        ) from exc


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
        raise PredictionServiceError(
            f"PredictionService connection failed: {exc}"
        ) from exc

    if response.status_code != 200:
        raise PredictionServiceError(
            f"PredictionService returned "
            f"status={response.status_code} "
            f"body={response.text}"
        )

    try:
        return response.json()

    except json.JSONDecodeError as exc:
        raise PredictionServiceError(
            "PredictionService returned invalid JSON."
        ) from exc


def health_check() -> Dict[str, Any]:
    return _get(
        url=HEALTH_ENDPOINT,
    )


def get_registered_models() -> Dict[str, Any]:
    return _get(
        url=MODELS_ENDPOINT,
    )


def predict_all(
    features: Dict[str, Any],
    *,
    request_source: str = "GUIService",
) -> Dict[str, Any]:
    payload = {
        "features": features,
        "request_source": request_source,
    }

    return _post(
        url=PREDICT_ALL_ENDPOINT,
        payload=payload,
    )


def explain_prediction(
    *,
    target_name: str,
    features: Dict[str, Any],
    top_n: int = 10,
    request_source: str = "GUIService",
) -> Dict[str, Any]:
    payload = {
        "target_name": target_name,
        "features": features,
        "top_n": top_n,
        "request_source": request_source,
    }

    return _post(
        url=EXPLAIN_ENDPOINT,
        payload=payload,
    )


def predict_and_explain_all(
    *,
    features: Dict[str, Any],
    explain_top_n: int = 5,
    request_source: str = "GUIService",
) -> Dict[str, Any]:
    prediction_response = predict_all(
        features=features,
        request_source=request_source,
    )

    predictions = prediction_response.get("predictions", {})

    explainability_results: Dict[str, Any] = {}

    for target_name in predictions.keys():
        try:
            explainability_results[target_name] = explain_prediction(
                target_name=target_name,
                features=features,
                top_n=explain_top_n,
                request_source=request_source,
            )

        except Exception as exc:
            explainability_results[target_name] = {
                "error": str(exc),
            }

    return {
        "predictions": predictions,
        "explanations": explainability_results,
    }