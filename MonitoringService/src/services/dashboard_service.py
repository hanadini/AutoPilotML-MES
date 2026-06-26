from __future__ import annotations

from sqlalchemy.orm import Session

from schemas.dashboard_schema import DashboardResponse

from services.monitoring_service import (
    build_system_overview,
    build_prediction_stats,
    build_feedback_stats,
)

from services.performance_service import (
    build_model_performance,
    build_model_performance_by_target,
)

from services.drift_service import (
    calculate_drift_status,
)

from services.retraining_service import (
    build_retraining_advisor,
)


def get_dashboard(
    db: Session,
) -> DashboardResponse:
    return DashboardResponse(
        system_overview=build_system_overview(db),
        prediction_stats=build_prediction_stats(db),
        feedback_stats=build_feedback_stats(db),
        model_performance=build_model_performance(db),
        model_performance_by_target=build_model_performance_by_target(db),
        drift_status=calculate_drift_status(db),
        retraining_advisor=build_retraining_advisor(db),
    )