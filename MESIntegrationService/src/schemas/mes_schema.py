from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MESPredictionRequest(BaseModel):
    production_id: str = Field(..., example="MDF1-2026-001")
    timestamp: Optional[datetime] = None
    line: Optional[str] = Field(default="MDF1")
    target: str = Field(..., example="labDensityAverage")
    features: Dict[str, Any]


class MESPredictionResponse(BaseModel):
    production_id: str
    target: str
    prediction: float
    risk_level: str
    recommendation: str
    top_influencing_features: List[str]

    model_name: Optional[str] = None
    algorithm_name: Optional[str] = None
    serving_type: Optional[str] = None
    shap_details: Optional[List[Dict[str, Any]]] = None