from __future__ import annotations

from sqlalchemy.orm import Session

from db.performance_repository import get_model_performance
from schemas.retraining_schema import RetrainingAdvisorResponse


MIN_RECORDS_FOR_DECISION = 10
WARNING_MAPE_THRESHOLD = 3.0
RETRAINING_MAPE_THRESHOLD = 5.0


def build_retraining_advisor(
    db: Session,
) -> RetrainingAdvisorResponse:
    performance = get_model_performance(db)

    total_records = performance.total_records
    current_mape = performance.mape

    if total_records < MIN_RECORDS_FOR_DECISION:
        return RetrainingAdvisorResponse(
            status="NOT_ENOUGH_DATA",
            total_records=total_records,
            current_mape=current_mape,
            warning_threshold=WARNING_MAPE_THRESHOLD,
            retraining_threshold=RETRAINING_MAPE_THRESHOLD,
            message=(
                "Not enough quality result records are available "
                "to make a retraining decision."
            ),
        )

    if current_mape >= RETRAINING_MAPE_THRESHOLD:
        return RetrainingAdvisorResponse(
            status="RETRAINING_RECOMMENDED",
            total_records=total_records,
            current_mape=current_mape,
            warning_threshold=WARNING_MAPE_THRESHOLD,
            retraining_threshold=RETRAINING_MAPE_THRESHOLD,
            message=(
                "Model error exceeded the retraining threshold. "
                "Review recent production data and consider retraining."
            ),
        )

    if current_mape >= WARNING_MAPE_THRESHOLD:
        return RetrainingAdvisorResponse(
            status="WATCH",
            total_records=total_records,
            current_mape=current_mape,
            warning_threshold=WARNING_MAPE_THRESHOLD,
            retraining_threshold=RETRAINING_MAPE_THRESHOLD,
            message=(
                "Model error is increasing and should be monitored closely."
            ),
        )

    return RetrainingAdvisorResponse(
        status="HEALTHY",
        total_records=total_records,
        current_mape=current_mape,
        warning_threshold=WARNING_MAPE_THRESHOLD,
        retraining_threshold=RETRAINING_MAPE_THRESHOLD,
        message="Model performance is currently within the acceptable range.",
    )