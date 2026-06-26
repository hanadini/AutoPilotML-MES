from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from db.models import MESPredictionLog, OperatorFeedback, QualityResult


def get_prediction_count(
    db: Session,
) -> int:
    return db.query(MESPredictionLog).count()


def get_feedback_count(
    db: Session,
) -> int:
    return db.query(OperatorFeedback).count()


def get_quality_result_count(
    db: Session,
) -> int:
    return db.query(QualityResult).count()

def get_prediction_coverage(
    db: Session,
) -> float:
    total_predictions = get_prediction_count(db)
    total_quality_results = get_quality_result_count(db)

    if total_predictions == 0:
        return 0.0

    return round(
        (total_quality_results / total_predictions) * 100,
        2,
    )


def get_risk_distribution(
    db: Session,
) -> Dict[str, int]:
    rows = (
        db.query(
            MESPredictionLog.risk_level,
            func.count(MESPredictionLog.id),
        )
        .group_by(MESPredictionLog.risk_level)
        .all()
    )

    return {
        str(risk_level): int(count)
        for risk_level, count in rows
    }


def get_feedback_distribution(
    db: Session,
) -> Dict[str, int]:
    rows = (
        db.query(
            OperatorFeedback.operator_decision,
            func.count(OperatorFeedback.id),
        )
        .group_by(OperatorFeedback.operator_decision)
        .all()
    )

    return {
        str(decision): int(count)
        for decision, count in rows
    }


def get_recommendation_acceptance_rate(
    db: Session,
) -> float:
    total = db.query(OperatorFeedback).count()

    if total == 0:
        return 0.0

    accepted = (
        db.query(OperatorFeedback)
        .filter(OperatorFeedback.operator_decision == "ACCEPTED")
        .count()
    )

    return round((accepted / total) * 100, 2)


def get_system_overview(
    db: Session,
) -> Dict[str, Any]:
    return {
    "total_mes_predictions": get_prediction_count(db),
    "total_quality_results": get_quality_result_count(db),
    "prediction_coverage_percent": get_prediction_coverage(db),
    "total_operator_feedback": get_feedback_count(db),
    "risk_distribution": get_risk_distribution(db),
    "feedback_distribution": get_feedback_distribution(db),
    "recommendation_acceptance_rate_percent": (
        get_recommendation_acceptance_rate(db)
        ),
    }