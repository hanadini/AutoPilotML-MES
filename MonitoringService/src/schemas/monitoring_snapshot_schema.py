from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class MonitoringSnapshotResponse(BaseModel):
    id: int
    snapshot_time: Optional[datetime] = None

    total_predictions: int
    total_quality_results: int
    coverage_percent: float

    mae: float
    mape: float

    drift_score: float
    drift_status: str

    retraining_status: str


class MonitoringSnapshotListResponse(BaseModel):
    items: List[MonitoringSnapshotResponse]