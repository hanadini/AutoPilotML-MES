from __future__ import annotations

from typing import Any, Dict

from sqlalchemy.orm import Session

from db.monitoring_repository import (
    get_feedback_count,
    get_feedback_distribution,
    get_prediction_count,
    get_quality_result_count,
    get_recommendation_acceptance_rate,
    get_risk_distribution,
    get_system_overview,
)

from utils.logging_utils import get_logger


logger = get_logger(__name__)


def build_system_overview(
    db: Session,
) -> Dict[str, Any]:
    overview = get_system_overview(db)

    logger.info(
        "System overview generated | predictions=%s | quality_results=%s | feedback=%s",
        overview.get("total_mes_predictions"),
        overview.get("total_quality_results"),
        overview.get("total_operator_feedback"),
    )

    return overview


def build_prediction_stats(
    db: Session,
) -> Dict[str, Any]:
    return {
        "total_mes_predictions": get_prediction_count(db),
        "total_quality_results": get_quality_result_count(db),
        "risk_distribution": get_risk_distribution(db),
    }


def build_feedback_stats(
    db: Session,
) -> Dict[str, Any]:
    return {
        "total_operator_feedback": get_feedback_count(db),
        "feedback_distribution": get_feedback_distribution(db),
        "recommendation_acceptance_rate_percent": (
            get_recommendation_acceptance_rate(db)
        ),
    }