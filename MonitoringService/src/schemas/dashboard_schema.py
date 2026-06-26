from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from schemas.drift_schema import DriftStatusResponse
from schemas.performance_schema import (
    ModelPerformanceResponse,
    TargetPerformanceResponse,
)
from schemas.retraining_schema import RetrainingAdvisorResponse


class DashboardResponse(BaseModel):
    system_overview: Dict[str, Any]
    prediction_stats: Dict[str, Any]
    feedback_stats: Dict[str, Any]

    model_performance: ModelPerformanceResponse
    model_performance_by_target: TargetPerformanceResponse

    drift_status: DriftStatusResponse
    retraining_advisor: RetrainingAdvisorResponse