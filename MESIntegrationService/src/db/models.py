from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from db.database import Base


class MESPredictionLog(Base):
    __tablename__ = "mes_prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    production_id = Column(String(100), index=True, nullable=False)
    line = Column(String(50), nullable=True)
    target = Column(String(100), index=True, nullable=False)

    prediction = Column(Float, nullable=False)
    risk_level = Column(String(20), index=True, nullable=False)

    model_name = Column(String(200), nullable=True)
    algorithm_name = Column(String(100), nullable=True)
    serving_type = Column(String(100), nullable=True)

    recommendation = Column(Text, nullable=True)
    top_influencing_features = Column(Text, nullable=True)
    shap_details = Column(Text, nullable=True)
    raw_mes_payload = Column(Text, nullable=True)

    mes_timestamp = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())