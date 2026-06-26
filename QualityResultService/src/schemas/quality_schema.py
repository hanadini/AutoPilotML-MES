from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# class QualityResultRequest(BaseModel):
#     production_id: str = Field(..., example="MDF1-MES-TEST-999")
#     target: str = Field(..., example="labDensityAverage")
#
#     predicted_value: Optional[float] = Field(default=None, example=712.4)
#     actual_value: float = Field(..., example=705.0)
#
#     lab_timestamp: Optional[datetime] = Field(
#         default=None,
#         example="2026-06-18T15:00:00",
#     )

class QualityResultRequest(BaseModel):
    production_id: str
    target: str
    actual_value: float
    lab_timestamp: Optional[datetime] = None


class QualityResultResponse(BaseModel):
    id: int
    prediction_log_id: Optional[int] = None

    production_id: str
    target: str
    predicted_value: Optional[float] = None
    actual_value: float
    absolute_error: Optional[float] = None
    percentage_error: Optional[float] = None
    lab_timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None