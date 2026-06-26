from __future__ import annotations

from typing import List

from pydantic import BaseModel


class ModelPerformanceResponse(BaseModel):
    total_records: int
    mae: float
    mape: float


class TargetPerformanceItem(BaseModel):
    target: str
    samples: int
    mae: float
    mape: float


class TargetPerformanceResponse(BaseModel):
    items: List[TargetPerformanceItem]