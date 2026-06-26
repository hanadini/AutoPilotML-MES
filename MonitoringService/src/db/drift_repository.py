from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.models import QualityResult


def get_quality_results_for_drift(
    db: Session,
    limit: int = 60,
) -> List[QualityResult]:
    return (
        db.query(QualityResult)
        .filter(QualityResult.percentage_error.isnot(None))
        .order_by(QualityResult.created_at.desc())
        .limit(limit)
        .all()
    )