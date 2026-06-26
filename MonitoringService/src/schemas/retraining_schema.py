from __future__ import annotations

from pydantic import BaseModel


class RetrainingAdvisorResponse(BaseModel):
    status: str
    total_records: int
    current_mape: float
    warning_threshold: float
    retraining_threshold: float
    message: str