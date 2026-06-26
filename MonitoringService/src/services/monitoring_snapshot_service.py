from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.monitoring_repository import (
    get_prediction_count,
    get_prediction_coverage,
    get_quality_result_count,
)
from db.monitoring_snapshot_repository import (
    get_latest_monitoring_snapshots,
    save_monitoring_snapshot,
)
from schemas.monitoring_snapshot_schema import (
    MonitoringSnapshotListResponse,
    MonitoringSnapshotResponse,
)
from services.drift_service import calculate_drift_status
from services.performance_service import build_model_performance
from services.retraining_service import build_retraining_advisor
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def create_monitoring_snapshot(
    db: Session,
) -> MonitoringSnapshotResponse:
    performance = build_model_performance(db)
    drift_status = calculate_drift_status(db)
    retraining_advisor = build_retraining_advisor(db)

    response = save_monitoring_snapshot(
        db=db,
        total_predictions=get_prediction_count(db),
        total_quality_results=get_quality_result_count(db),
        coverage_percent=get_prediction_coverage(db),
        mae=performance.mae,
        mape=performance.mape,
        drift_score=drift_status.drift_score,
        drift_status=drift_status.status,
        retraining_status=retraining_advisor.status,
    )

    logger.info(
        "Monitoring snapshot saved | id=%s | coverage=%s | mae=%s | mape=%s | drift=%s",
        response.id,
        response.coverage_percent,
        response.mae,
        response.mape,
        response.drift_status,
    )

    return response


def list_monitoring_snapshots(
    db: Session,
    limit: int = 4,
) -> MonitoringSnapshotListResponse:
    return MonitoringSnapshotListResponse(
        items=get_latest_monitoring_snapshots(
            db=db,
            limit=limit,
        )
    )