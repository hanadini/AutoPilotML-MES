from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.sql import func

from db.database import Base


class MESPredictionLog(Base):
    __tablename__ = "mes_prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    production_id = Column(String(100), nullable=False, index=True)
    line = Column(String(50), nullable=True)
    target = Column(String(100), nullable=False, index=True)

    prediction = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=True)

    model_name = Column(String(200), nullable=True)
    algorithm_name = Column(String(100), nullable=True)
    serving_type = Column(String(100), nullable=True)

    recommendation = Column(Text, nullable=True)
    top_influencing_features = Column(Text, nullable=True)
    shap_details = Column(Text, nullable=True)
    raw_mes_payload = Column(Text, nullable=True)

    mes_timestamp = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class QualityResult(Base):
    __tablename__ = "quality_results"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    prediction_log_id = Column(
        Integer,
        ForeignKey("mes_prediction_logs.id"),
        nullable=True,
        index=True,
    )

    production_id = Column(
        String(100),
        nullable=False,
        index=True,
    )

    target = Column(
        String(100),
        nullable=False,
        index=True,
    )

    predicted_value = Column(
        Float,
        nullable=True,
    )

    actual_value = Column(
        Float,
        nullable=False,
    )

    absolute_error = Column(
        Float,
        nullable=True,
    )

    percentage_error = Column(
        Float,
        nullable=True,
    )

    lab_timestamp = Column(
        DateTime,
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )