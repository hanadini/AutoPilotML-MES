from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import MESPredictionLog, QualityResult
from schemas.quality_schema import (
    QualityResultRequest,
    QualityResultResponse,
)


def _calculate_absolute_error(
    predicted_value: Optional[float],
    actual_value: float,
) -> Optional[float]:
    if predicted_value is None:
        return None

    return round(
        abs(predicted_value - actual_value),
        4,
    )


def _calculate_percentage_error(
    predicted_value: Optional[float],
    actual_value: float,
) -> Optional[float]:
    if predicted_value is None:
        return None

    if actual_value == 0:
        return None

    return round(
        abs(predicted_value - actual_value) / abs(actual_value) * 100,
        2,
    )


def _find_matching_prediction_log(
    db: Session,
    production_id: str,
    target: str,
) -> Optional[MESPredictionLog]:
    return (
        db.query(MESPredictionLog)
        .filter(MESPredictionLog.production_id == production_id)
        .filter(MESPredictionLog.target == target)
        .order_by(MESPredictionLog.created_at.desc())
        .first()
    )


def _quality_result_exists(
    db: Session,
    production_id: str,
    target: str,
) -> bool:
    return (
        db.query(QualityResult)
        .filter(QualityResult.production_id == production_id)
        .filter(QualityResult.target == target)
        .first()
        is not None
    )


def save_quality_result(
    db: Session,
    request: QualityResultRequest,
) -> QualityResultResponse:
    prediction_log = _find_matching_prediction_log(
        db=db,
        production_id=request.production_id,
        target=request.target,
    )

    if prediction_log is None:
        raise ValueError(
            "No matching MES prediction log found for "
            f"production_id={request.production_id}, "
            f"target={request.target}."
        )

    if _quality_result_exists(
            db=db,
            production_id=request.production_id,
            target=request.target,
    ):
        raise ValueError(
            "Quality result already exists for "
            f"production_id={request.production_id}, "
            f"target={request.target}."
        )

    prediction_log_id = prediction_log.id
    predicted_value = prediction_log.prediction

    absolute_error = _calculate_absolute_error(
        predicted_value=predicted_value,
        actual_value=request.actual_value,
    )

    percentage_error = _calculate_percentage_error(
        predicted_value=predicted_value,
        actual_value=request.actual_value,
    )

    record = QualityResult(
        prediction_log_id=prediction_log_id,
        production_id=request.production_id,
        target=request.target,
        predicted_value=predicted_value,
        actual_value=request.actual_value,
        absolute_error=absolute_error,
        percentage_error=percentage_error,
        lab_timestamp=request.lab_timestamp,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return QualityResultResponse(
        id=record.id,
        prediction_log_id=record.prediction_log_id,
        production_id=record.production_id,
        target=record.target,
        predicted_value=record.predicted_value,
        actual_value=record.actual_value,
        absolute_error=record.absolute_error,
        percentage_error=record.percentage_error,
        lab_timestamp=record.lab_timestamp,
        created_at=record.created_at,
    )


def get_latest_quality_results(
    db: Session,
    limit: int = 20,
) -> List[QualityResultResponse]:
    rows = (
        db.query(QualityResult)
        .order_by(QualityResult.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        QualityResultResponse(
            id=row.id,
            prediction_log_id=row.prediction_log_id,
            production_id=row.production_id,
            target=row.target,
            predicted_value=row.predicted_value,
            actual_value=row.actual_value,
            absolute_error=row.absolute_error,
            percentage_error=row.percentage_error,
            lab_timestamp=row.lab_timestamp,
            created_at=row.created_at,
        )
        for row in rows
    ]