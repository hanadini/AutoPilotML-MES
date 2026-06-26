from __future__ import annotations

from typing import List

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import QualityResult
from schemas.performance_schema import (
    ModelPerformanceResponse,
    TargetPerformanceItem,
)


def get_model_performance(
    db: Session,
) -> ModelPerformanceResponse:
    row = (
        db.query(
            func.count(QualityResult.id),
            func.avg(QualityResult.absolute_error),
            func.avg(QualityResult.percentage_error),
        )
        .one()
    )

    total_records = int(row[0] or 0)
    mae = round(float(row[1] or 0.0), 4)
    mape = round(float(row[2] or 0.0), 4)

    return ModelPerformanceResponse(
        total_records=total_records,
        mae=mae,
        mape=mape,
    )


def get_model_performance_by_target(
    db: Session,
) -> List[TargetPerformanceItem]:
    rows = (
        db.query(
            QualityResult.target,
            func.count(QualityResult.id),
            func.avg(QualityResult.absolute_error),
            func.avg(QualityResult.percentage_error),
        )
        .group_by(QualityResult.target)
        .order_by(QualityResult.target)
        .all()
    )

    return [
        TargetPerformanceItem(
            target=str(target),
            samples=int(samples or 0),
            mae=round(float(mae or 0.0), 4),
            mape=round(float(mape or 0.0), 4),
        )
        for target, samples, mae, mape in rows
    ]