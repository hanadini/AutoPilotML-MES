from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.feedback_repository import (
    get_latest_operator_feedback,
    save_operator_feedback,
)
from schemas.feedback_schema import (
    OperatorFeedbackRequest,
    OperatorFeedbackResponse,
)
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def create_feedback(
    db: Session,
    request: OperatorFeedbackRequest,
) -> OperatorFeedbackResponse:
    response = save_operator_feedback(
        db=db,
        request=request,
    )

    logger.info(
        "Operator feedback saved | production_id=%s | target=%s | decision=%s",
        response.production_id,
        response.target,
        response.operator_decision,
    )

    return response


def list_feedback(
    db: Session,
    limit: int = 20,
) -> List[OperatorFeedbackResponse]:
    return get_latest_operator_feedback(
        db=db,
        limit=limit,
    )