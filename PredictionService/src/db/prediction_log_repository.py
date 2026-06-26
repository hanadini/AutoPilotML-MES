from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import PredictionLog
from utils.decorators import service_step
from utils.logging_utils import get_logger


logger = get_logger(__name__)


@service_step
def save_prediction_log(
    db: Session,
    target_name: str,
    prediction_value: Optional[float],
    model_name: Optional[str],
    algorithm_name: Optional[str],
    input_features: Dict[str, Any],
    missing_features: List[str],
    extra_features: Optional[List[str]] = None,
    invalid_numeric_features: Optional[List[str]] = None,
    null_features: Optional[List[str]] = None,
    forced_missing_features: Optional[List[str]] = None,
    serving_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    request_source: str = "PredictionService",
    production_order: Optional[str] = None,
    board_id: Optional[str] = None,
    status: str = "SUCCESS",
    error_message: Optional[str] = None,
) -> Optional[PredictionLog]:
    try:
        log = PredictionLog(
            target_name=target_name,
            prediction_value=prediction_value,
            model_name=model_name,
            algorithm_name=algorithm_name,
            serving_type=serving_type,
            risk_level=risk_level,
            input_features=input_features,
            missing_features=missing_features,
            extra_features=extra_features or [],
            invalid_numeric_features=invalid_numeric_features or [],
            null_features=null_features or [],
            forced_missing_features=forced_missing_features or [],
            request_source=request_source,
            production_order=production_order,
            board_id=board_id,
            status=status,
            error_message=error_message,
        )

        db.add(log)
        db.commit()
        db.refresh(log)

        logger.info(
            "Prediction log saved | target=%s | model=%s | risk=%s",
            target_name,
            model_name,
            risk_level,
        )

        return log

    except Exception:
        db.rollback()

        logger.exception(
            "Failed to save prediction log | target=%s | prediction will still be returned",
            target_name,
        )

        return None