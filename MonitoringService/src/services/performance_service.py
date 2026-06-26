from __future__ import annotations

from sqlalchemy.orm import Session

from db.performance_repository import (
    get_model_performance,
    get_model_performance_by_target,
)
from schemas.performance_schema import (
    ModelPerformanceResponse,
    TargetPerformanceResponse,
)


def build_model_performance(
    db: Session,
) -> ModelPerformanceResponse:
    return get_model_performance(db)


def build_model_performance_by_target(
    db: Session,
) -> TargetPerformanceResponse:
    return TargetPerformanceResponse(
        items=get_model_performance_by_target(db)
    )