from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.dependencies import get_db
from schemas.quality_schema import (
    QualityResultRequest,
    QualityResultResponse,
)
from services.quality_service import (
    create_quality_result,
    list_quality_results,
)


router = APIRouter(
    prefix="/api/v1/quality-results",
    tags=["Quality Results"],
)


@router.post(
    "",
    response_model=QualityResultResponse,
)
def submit_quality_result(
    request: QualityResultRequest,
    db: Session = Depends(get_db),
) -> QualityResultResponse:
    return create_quality_result(
        db=db,
        request=request,
    )


@router.get(
    "",
    response_model=List[QualityResultResponse],
)
def get_quality_results(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> List[QualityResultResponse]:
    return list_quality_results(
        db=db,
        limit=limit,
    )