from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from db.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(
        BigInteger,
        primary_key=True,
        index=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    target_name = Column(
        String(100),
        nullable=False,
        index=True,
    )

    prediction_value = Column(
        Float,
        nullable=True,
    )

    model_name = Column(
        String(200),
        nullable=True,
    )

    algorithm_name = Column(
        String(100),
        nullable=True,
    )

    serving_type = Column(
        String(100),
        nullable=True,
    )

    risk_level = Column(
        String(50),
        nullable=True,
        index=True,
    )

    input_features = Column(
        JSONB,
        nullable=True,
    )

    missing_features = Column(
        JSONB,
        nullable=True,
    )

    extra_features = Column(
        JSONB,
        nullable=True,
    )

    invalid_numeric_features = Column(
        JSONB,
        nullable=True,
    )

    null_features = Column(
        JSONB,
        nullable=True,
    )

    forced_missing_features = Column(
        JSONB,
        nullable=True,
    )

    request_source = Column(
        String(100),
        nullable=True,
        index=True,
    )

    production_order = Column(
        String(100),
        nullable=True,
        index=True,
    )

    board_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    status = Column(
        String(50),
        nullable=False,
        default="SUCCESS",
        index=True,
    )

    error_message = Column(
        Text,
        nullable=True,
    )