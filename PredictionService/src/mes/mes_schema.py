from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from db.database import Base


class MESPredictionRequest(BaseModel):
    production_id: str
    timestamp: Optional[datetime] = None
    line: Optional[str] = "MDF1"
    target: str
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

