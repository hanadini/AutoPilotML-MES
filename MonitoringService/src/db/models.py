from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from db.database import Base


class MESPredictionLog(Base):
    __tablename__ = "mes_prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    production_id = Column(String(100), nullable=False, index=True)
    target = Column(String(100), nullable=False, index=True)
    prediction = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class QualityResult(Base):
    __tablename__ = "quality_results"

    id = Column(Integer, primary_key=True, index=True)

    prediction_log_id = Column(
        Integer,
        ForeignKey("mes_prediction_logs.id"),
        nullable=True,
        index=True,
    )

    production_id = Column(String(100), nullable=False, index=True)
    target = Column(String(100), nullable=False, index=True)

    predicted_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=False)

    absolute_error = Column(Float, nullable=True)
    percentage_error = Column(Float, nullable=True)

    lab_timestamp = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class OperatorFeedback(Base):
    __tablename__ = "operator_feedback"

    id = Column(Integer, primary_key=True, index=True)

    production_id = Column(String(100), nullable=False, index=True)
    target = Column(String(100), nullable=False, index=True)

    prediction_value = Column(String(100), nullable=True)
    risk_level = Column(String(50), nullable=True, index=True)
    recommendation = Column(String, nullable=True)

    operator_decision = Column(String(50), nullable=False, index=True)
    operator_comment = Column(String, nullable=True)
    operator_name = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MonitoringSnapshot(Base):
    __tablename__ = "monitoring_snapshots"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    snapshot_time = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    total_predictions = Column(
        Integer,
        nullable=False,
        default=0,
    )

    total_quality_results = Column(
        Integer,
        nullable=False,
        default=0,
    )

    coverage_percent = Column(
        Float,
        nullable=False,
        default=0.0,
    )

    mae = Column(
        Float,
        nullable=False,
        default=0.0,
    )

    mape = Column(
        Float,
        nullable=False,
        default=0.0,
    )

    drift_score = Column(
        Float,
        nullable=False,
        default=0.0,
    )

    drift_status = Column(
        String(50),
        nullable=False,
        default="UNKNOWN",
    )

    retraining_status = Column(
        String(50),
        nullable=False,
        default="UNKNOWN",
    )