from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.shared_model_registry import shared_model_registry
from db.dependencies import get_db
from db.prediction_log_repository import save_prediction_log
from schemas.request_schema import PredictAllRequest
from schemas.response_schema import (
    PredictAllItemResponse,
    PredictAllResponse,
)
from service.explanation_service import explain_prediction
from service.feature_metadata import load_important_features_by_target
from utils.logging_utils import get_logger


logger = get_logger(__name__)

router = APIRouter()
model_registry = shared_model_registry


def _ensure_registry_loaded() -> None:
    if not model_registry.target_names():
        model_registry.load_all_from_registry()

    logger.info(
        "Loaded registry targets: %s",
        model_registry.target_names(),
    )


@router.post(
    "/predict-all",
    response_model=PredictAllResponse,
    summary="Predict all registered MDF1 targets",
)
def predict_all(
    request: PredictAllRequest,
    db: Session = Depends(get_db),
) -> PredictAllResponse:

    logger.info(
        "PREDICT_ALL_ENTERED | board_id=%s | feature_count=%s",
        request.board_id,
        len(request.features),
    )

    start_time = time.perf_counter()

    try:
        _ensure_registry_loaded()

        predictions: Dict[str, PredictAllItemResponse] = {}
        used_feature_count: Dict[str, int] = {}

        important_features = load_important_features_by_target(
            top_n=5,
        )

        logger.info(
            "Predict-all request received | board_id=%s | production_order=%s | request_source=%s | feature_count=%s",
            request.board_id,
            request.production_order,
            request.request_source,
            len(request.features),
        )

        logger.info(
            "Predict-all raw input check | pressPressure1C=%s | pressPressure10C=%s | pressPressure21C=%s | "
            "pressPressure1L=%s | pressPressure10L=%s | pressPressure21L=%s | "
            "pressPressure1R=%s | pressPressure10R=%s | pressPressure21R=%s | "
            "operPressTemp1=%s | pressHotPlateTemp1=%s | heatInletTemp=%s | operFibreTemp=%s | "
            "thicknessClosed1=%s | thicknessClosed4=%s | thicknessClosed7=%s",
            request.features.get("pressPressure1C"),
            request.features.get("pressPressure10C"),
            request.features.get("pressPressure21C"),
            request.features.get("pressPressure1L"),
            request.features.get("pressPressure10L"),
            request.features.get("pressPressure21L"),
            request.features.get("pressPressure1R"),
            request.features.get("pressPressure10R"),
            request.features.get("pressPressure21R"),
            request.features.get("operPressTemp1"),
            request.features.get("pressHotPlateTemp1"),
            request.features.get("heatInletTemp"),
            request.features.get("operFibreTemp"),
            request.features.get("thicknessClosed1"),
            request.features.get("thicknessClosed4"),
            request.features.get("thicknessClosed7"),
        )

        for target in model_registry.target_names():
            try:
                explanation_result: Dict[str, Any] = explain_prediction(
                    target=target,
                    incoming_features=request.features,
                    top_n=15,
                )

                prediction_value = explanation_result["prediction"]
                risk_level = explanation_result.get("risk_level")
                explanation = explanation_result.get("explanation", [])
                recommendations = explanation_result.get(
                    "recommendations",
                    [],
                )

                predictions[target] = PredictAllItemResponse(
                    model_name=explanation_result["model_name"],
                    algorithm_name=explanation_result["algorithm_name"],
                    prediction=float(prediction_value),
                    risk_level=risk_level,
                    explanation=explanation,
                    recommendations=recommendations,
                )

                used_feature_count[target] = explanation_result.get(
                    "used_feature_count",
                    0,
                )

                save_prediction_log(
                    db=db,
                    target_name=target,
                    prediction_value=float(prediction_value),
                    model_name=explanation_result["model_name"],
                    algorithm_name=explanation_result["algorithm_name"],
                    serving_type=explanation_result.get("serving_type"),
                    risk_level=risk_level,
                    input_features=request.features,
                    missing_features=explanation_result.get("missing_features", []),
                    extra_features=explanation_result.get("extra_features", []),
                    invalid_numeric_features=explanation_result.get(
                        "invalid_numeric_features",
                        [],
                    ),
                    null_features=explanation_result.get("null_features", []),
                    forced_missing_features=explanation_result.get(
                        "forced_missing_features",
                        [],
                    ),
                    request_source=request.request_source,
                    production_order=request.production_order,
                    board_id=request.board_id,
                )

                logger.info(
                    "Predict-all target completed | target=%s | prediction=%s | risk=%s",
                    target,
                    prediction_value,
                    risk_level,
                )

            except Exception as exc:
                logger.exception(
                    "Predict-all target failed | target=%s | error=%s",
                    target,
                    str(exc),
                )

                predictions[target] = PredictAllItemResponse(
                    model_name="FAILED",
                    algorithm_name="FAILED",
                    prediction=0.0,
                    risk_level="ERROR",
                    explanation=[],
                    recommendations=[
                        {
                            "feature": "system",
                            "feature_label": "System Validation",
                            "impact": 0.0,
                            "direction": "unknown",
                            "risk_level": "ERROR",
                            "message": str(exc),
                        }
                    ],
                )

                used_feature_count[target] = 0

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.info(
            "Predict-all completed | target_count=%s | duration_ms=%s",
            len(predictions),
            elapsed_ms,
        )

        return PredictAllResponse(
            predictions=predictions,
            used_feature_count_by_target=used_feature_count,
            important_features_by_target=important_features,
        )

    except FileNotFoundError as exc:
        logger.error(
            "Predict-all failed | file not found | detail=%s",
            str(exc),
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except (KeyError, ValueError) as exc:
        logger.warning(
            "Predict-all failed | validation error | detail=%s",
            str(exc),
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Predict-all failed | unexpected error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )