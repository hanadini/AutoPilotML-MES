from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.dependencies import get_db
from schemas.feedback_schema import (
    OperatorFeedbackRequest,
    OperatorFeedbackResponse,
)
from services.feedback_service import create_feedback, list_feedback


router = APIRouter(
    prefix="/api/v1/operator-feedback",
    tags=["Operator Feedback"],
)


@router.post(
    " Submit",
    response_model=OperatorFeedbackResponse,
)
def submit_feedback(
    request: OperatorFeedbackRequest,
    db: Session = Depends(get_db),
) -> OperatorFeedbackResponse:
    try:
        return create_feedback(
            db=db,
            request=request,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.get(
    " Recieve",
    response_model=List[OperatorFeedbackResponse],
)
def get_feedback(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> List[OperatorFeedbackResponse]:
    return list_feedback(
        db=db,
        limit=limit,
    )