from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OperatorFeedbackRequest(BaseModel):
    production_id: str = Field(..., example="MDF1-MES-TEST-999")
    target: str = Field(..., example="labDensityAverage")
    prediction_value: Optional[str] = Field(default=None, example="712.4")
    risk_level: Optional[str] = Field(default=None, example="MEDIUM")
    recommendation: Optional[str] = Field(
        default=None,
        example="Review belt speed and press fibre density.",
    )
    operator_decision: str = Field(
        ...,
        example="ACCEPTED",
        description="ACCEPTED, REJECTED, IGNORED, or MODIFIED",
    )
    operator_comment: Optional[str] = Field(
        default=None,
        example="Applied recommendation during shift B.",
    )
    operator_name: Optional[str] = Field(default=None, example="Operator A")


class OperatorFeedbackResponse(BaseModel):
    id: int
    production_id: str
    target: str
    prediction_value: Optional[str] = None
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None
    operator_decision: str
    operator_comment: Optional[str] = None
    operator_name: Optional[str] = None
    created_at: Optional[datetime] = None