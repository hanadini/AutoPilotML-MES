from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.models import OperatorFeedback
from schemas.feedback_schema import (
    OperatorFeedbackRequest,
    OperatorFeedbackResponse,
)
from config.settings import VALID_DECISIONS


def save_operator_feedback(
    db: Session,
    request: OperatorFeedbackRequest,
) -> OperatorFeedbackResponse:
    decision = request.operator_decision.upper().strip()

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid operator_decision '{request.operator_decision}'. "
            f"Allowed values: {sorted(VALID_DECISIONS)}"
        )

    record = OperatorFeedback(
        production_id=request.production_id,
        target=request.target,
        prediction_value=request.prediction_value,
        risk_level=request.risk_level,
        recommendation=request.recommendation,
        operator_decision=decision,
        operator_comment=request.operator_comment,
        operator_name=request.operator_name,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return OperatorFeedbackResponse(
        id=record.id,
        production_id=record.production_id,
        target=record.target,
        prediction_value=record.prediction_value,
        risk_level=record.risk_level,
        recommendation=record.recommendation,
        operator_decision=record.operator_decision,
        operator_comment=record.operator_comment,
        operator_name=record.operator_name,
        created_at=record.created_at,
    )


def get_latest_operator_feedback(
    db: Session,
    limit: int = 20,
) -> List[OperatorFeedbackResponse]:
    rows = (
        db.query(OperatorFeedback)
        .order_by(OperatorFeedback.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        OperatorFeedbackResponse(
            id=row.id,
            production_id=row.production_id,
            target=row.target,
            prediction_value=row.prediction_value,
            risk_level=row.risk_level,
            recommendation=row.recommendation,
            operator_decision=row.operator_decision,
            operator_comment=row.operator_comment,
            operator_name=row.operator_name,
            created_at=row.created_at,
        )
        for row in rows
    ]