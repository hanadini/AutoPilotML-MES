from __future__ import annotations

import threading
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from core.shared_model_registry import shared_model_registry
from schemas.request_schema import PredictionRequest
from schemas.response_schema import PredictionResponse
from service.explanation_service import explain_prediction
from service.predictor import predict_single_model, predict_weighted_ensemble
from utils.logging_utils import get_logger


logger = get_logger(__name__)

router = APIRouter()

model_registry = shared_model_registry


# Lock threading:
# Has registry already been loaded?
# Yes → continue prediction
# No  → one request loads it, others wait

_registry_loaded = threading.Event()
_registry_lock = threading.Lock()


def _ensure_registry_loaded() -> None:
    if _registry_loaded.is_set():
        return

    with _registry_lock:
        if not _registry_loaded.is_set():
            logger.info("Loading model registry into memory")
            model_registry.load_all_from_registry()
            _registry_loaded.set()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    start_time = time.perf_counter()

    try:
        _ensure_registry_loaded()

        target = request.target.strip()

        logger.info(
            "Prediction request received | target=%s",
            target,
        )

        registry_entry = model_registry.get_registry_entry(target)

        if registry_entry.serving_type == "single_model":
            artifact = model_registry.get_artifact_for_target(target)

            result: Dict[str, Any] = predict_single_model(
                target_name=target,
                incoming_features=request.features,
                artifact=artifact,
            )

        elif registry_entry.serving_type == "weighted_ensemble":
            ensemble_members = model_registry.get_ensemble_members_for_target(
                target
            )

            result = predict_weighted_ensemble(
                target_name=target,
                incoming_features=request.features,
                ensemble_members=ensemble_members,
            )

        else:
            raise ValueError(
                f"Unsupported serving type: {registry_entry.serving_type}"
            )

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000
        )

        logger.info(
            "Prediction completed | duration_ms=%.2f",
            elapsed_ms,
        )

        result["duration_ms"] = elapsed_ms

        logger.info(
            "Prediction completed | target=%s | model=%s | algorithm=%s | duration_ms=%s",
            target,
            result.get("model_name"),
            result.get("algorithm_name"),
            elapsed_ms,
        )

        return PredictionResponse(**result)

    except FileNotFoundError as exc:
        logger.error(
            "Prediction failed | file not found | detail=%s",
            str(exc),
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except (KeyError, ValueError) as exc:
        logger.warning(
            "Prediction failed | validation error | detail=%s",
            str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Prediction failed | unexpected error"
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post("/explain", summary="Explain single-target prediction")
def explain(request: PredictionRequest) -> Dict[str, Any]:
    start_time = time.perf_counter()

    try:
        _ensure_registry_loaded()

        target = request.target.strip()

        logger.info(
            "Explanation request received | target=%s",
            target,
        )

        result = explain_prediction(
            target=target,
            incoming_features=request.features,
            top_n=15,
        )

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000.0)


        result["duration_ms"] = elapsed_ms

        logger.info(
            "Explanation completed | target=%s | model=%s | algorithm=%s | duration_ms=%s",
            target,
            result.get("model_name"),
            result.get("algorithm_name"),
            elapsed_ms,
        )

        return result

    except FileNotFoundError as exc:
        logger.error(
            "Explanation failed | file not found | detail=%s",
            str(exc),
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except (KeyError, ValueError) as exc:
        logger.warning(
            "Explanation failed | validation error | detail=%s",
            str(exc),
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Explanation failed | unexpected error"
        )
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )