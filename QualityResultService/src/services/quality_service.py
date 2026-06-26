from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.quality_repository import (
    get_latest_quality_results,
    save_quality_result,
)
from schemas.quality_schema import (
    QualityResultRequest,
    QualityResultResponse,
)
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def create_quality_result(
    db: Session,
    request: QualityResultRequest,
) -> QualityResultResponse:
    response = save_quality_result(
        db=db,
        request=request,
    )

    logger.info(
        "Quality result saved | production_id=%s | target=%s | actual=%s | abs_error=%s",
        response.production_id,
        response.target,
        response.actual_value,
        response.absolute_error,
    )

    return response


def list_quality_results(
    db: Session,
    limit: int = 20,
) -> List[QualityResultResponse]:
    return get_latest_quality_results(
        db=db,
        limit=limit,
    )