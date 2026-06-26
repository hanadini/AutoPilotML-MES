from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.drift_repository import get_quality_results_for_drift
from db.models import QualityResult
from schemas.drift_schema import DriftStatusResponse


MIN_RECORDS_FOR_DRIFT = 20
DRIFT_WARNING_THRESHOLD = 1.0
DRIFT_DETECTED_THRESHOLD = 3.0


def _average_mape(
    records: List[QualityResult],
) -> float:
    values = [
        float(record.percentage_error)
        for record in records
        if record.percentage_error is not None
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_drift_status(
    db: Session,
) -> DriftStatusResponse:
    records = get_quality_results_for_drift(
        db=db,
        limit=60,
    )

    records = list(reversed(records))

    if len(records) < MIN_RECORDS_FOR_DRIFT:
        return DriftStatusResponse(
            status="NOT_ENOUGH_DATA",
            drift_score=0.0,
            current_mape=0.0,
            previous_mape=0.0,
            current_samples=len(records),
            previous_samples=0,
            message=(
                "Not enough quality result records are available "
                "to evaluate model drift."
            ),
        )

    midpoint = len(records) // 2

    previous_window = records[:midpoint]
    current_window = records[midpoint:]

    previous_mape = _average_mape(previous_window)
    current_mape = _average_mape(current_window)

    drift_score = current_mape - previous_mape

    if drift_score >= DRIFT_DETECTED_THRESHOLD:
        status = "DRIFT_DETECTED"
        message = (
            "Model drift detected. Current prediction error is significantly "
            "higher than the previous quality result window."
        )

    elif drift_score >= DRIFT_WARNING_THRESHOLD:
        status = "WARNING"
        message = (
            "Prediction error is increasing. Monitor model performance closely."
        )

    else:
        status = "HEALTHY"
        message = (
            "No significant drift detected based on available quality results."
        )

    return DriftStatusResponse(
        status=status,
        drift_score=round(drift_score, 4),
        current_mape=round(current_mape, 4),
        previous_mape=round(previous_mape, 4),
        current_samples=len(current_window),
        previous_samples=len(previous_window),
        message=message,
    )