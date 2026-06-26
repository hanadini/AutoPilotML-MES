from __future__ import annotations

from typing import Any, Dict

import requests
from fastapi import HTTPException

from config.settings import PREDICTION_SERVICE_URL
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def call_prediction_service(
    target: str,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    url = f"{PREDICTION_SERVICE_URL}/explain"

    payload = {
        "target": target,
        "features": features,
    }

    try:
        session = requests.Session()
        session.trust_env = False

        response = session.post(
            url,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        logger.info(
            "PredictionService call successful | target=%s | status=%s",
            target,
            response.status_code,
        )

        return response.json()

    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail=f"PredictionService timeout: {str(exc)}",
        )

    except requests.exceptions.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"PredictionService connection failed: {str(exc)}",
        )

    except requests.exceptions.HTTPError as exc:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"PredictionService returned error: {response.text}",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected PredictionService client error: {str(exc)}",
        )


def check_prediction_service_health() -> Dict[str, Any]:
    url = f"{PREDICTION_SERVICE_URL}/health"

    try:
        session = requests.Session()
        session.trust_env = False

        response = session.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"PredictionService health check failed: {str(exc)}",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected health check error: {str(exc)}",
        )