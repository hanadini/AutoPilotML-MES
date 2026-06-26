from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.sql import func

from db.database import Base


class OperatorFeedback(Base):
    __tablename__ = "operator_feedback"

    id = Column(
        Integer,
        primary_key=True,
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

    prediction_value = Column(
        String(100),
        nullable=True,
    )

    risk_level = Column(
        String(50),
        nullable=True,
        index=True,
    )

    recommendation = Column(
        Text,
        nullable=True,
    )

    operator_decision = Column(
        String(50),
        nullable=False,
        index=True,
    )

    operator_comment = Column(
        Text,
        nullable=True,
    )

    operator_name = Column(
        String(100),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )