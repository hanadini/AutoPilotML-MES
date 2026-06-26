from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.dependencies import get_db
from services.monitoring_service import (
    build_feedback_stats,
    build_prediction_stats,
    build_system_overview,
)

from schemas.performance_schema import (
    ModelPerformanceResponse,
    TargetPerformanceResponse,
)
from services.performance_service import (
    build_model_performance,
    build_model_performance_by_target,
)

from schemas.monitoring_snapshot_schema import (
    MonitoringSnapshotListResponse,
    MonitoringSnapshotResponse,
)
from services.monitoring_snapshot_service import (
    create_monitoring_snapshot,
    list_monitoring_snapshots,
)

from schemas.retraining_schema import RetrainingAdvisorResponse
from services.retraining_service import build_retraining_advisor
from schemas.drift_schema import DriftStatusResponse

from services.drift_service import (
    calculate_drift_status,
)

from schemas.dashboard_schema import DashboardResponse
from services.dashboard_service import get_dashboard

from fastapi import HTTPException
from utils.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/monitoring",
    tags=["Monitoring"],
)

@router.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
):
    try:
        return get_dashboard(db)

    except Exception as exc:
        logger.exception("Dashboard endpoint failed.")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

@router.get("/system-overview")
def system_overview(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return build_system_overview(db)


@router.get("/prediction-stats")
def prediction_stats(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return build_prediction_stats(db)


@router.get("/feedback-stats")
def feedback_stats(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    return build_feedback_stats(db)


@router.get(
    "/model-performance",
    response_model=ModelPerformanceResponse,
)
def model_performance(
    db: Session = Depends(get_db),
) -> ModelPerformanceResponse:
    return build_model_performance(db)


@router.get(
    "/model-performance-by-target",
    response_model=TargetPerformanceResponse,
)
def model_performance_by_target(
    db: Session = Depends(get_db),
) -> TargetPerformanceResponse:
    return build_model_performance_by_target(db)

@router.get(
    "/retraining-advisor",
    response_model=RetrainingAdvisorResponse,
)
def retraining_advisor(
    db: Session = Depends(get_db),
) -> RetrainingAdvisorResponse:
    return build_retraining_advisor(db)

@router.get(
    "/drift-status",
    response_model=DriftStatusResponse,
)
def drift_status(
    db: Session = Depends(get_db),
) -> DriftStatusResponse:
    return calculate_drift_status(db)


@router.post(
    "/snapshots",
    response_model=MonitoringSnapshotResponse,
)
def create_snapshot(
    db: Session = Depends(get_db),
) -> MonitoringSnapshotResponse:
    return create_monitoring_snapshot(db)


@router.get(
    "/snapshots",
    response_model=MonitoringSnapshotListResponse,
)
def get_snapshots(
    limit: int = 4,
    db: Session = Depends(get_db),
) -> MonitoringSnapshotListResponse:
    return list_monitoring_snapshots(
        db=db,
        limit=limit,
    )