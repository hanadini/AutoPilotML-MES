from __future__ import annotations

import json
from typing import Any, Dict, List

from db.database import SessionLocal
from db.models import MESPredictionLog
from schemas.mes_schema import MESPredictionRequest, MESPredictionResponse
from utils.logging_utils import get_logger


logger = get_logger(__name__)


def save_mes_prediction_log(
    request: MESPredictionRequest,
    response: MESPredictionResponse,
) -> None:
    db = SessionLocal()

    try:
        log_record = MESPredictionLog(
            production_id=request.production_id,
            line=request.line,
            target=response.target,
            prediction=response.prediction,
            risk_level=response.risk_level,
            model_name=response.model_name,
            algorithm_name=response.algorithm_name,
            serving_type=response.serving_type,
            recommendation=response.recommendation,
            top_influencing_features=json.dumps(
                response.top_influencing_features,
                ensure_ascii=False,
            ),
            shap_details=json.dumps(
                response.shap_details,
                ensure_ascii=False,
                default=str,
            ),
            raw_mes_payload=json.dumps(
                request.model_dump(),
                ensure_ascii=False,
                default=str,
            ),
            mes_timestamp=request.timestamp,
        )

        db.add(log_record)
        db.commit()

        logger.info(
            "MES prediction log saved | production_id=%s | target=%s | risk=%s",
            request.production_id,
            response.target,
            response.risk_level,
        )

    except Exception:
        db.rollback()
        logger.exception(
            "Failed to save MES prediction log | production_id=%s",
            request.production_id,
        )

    finally:
        db.close()


def get_latest_mes_prediction_logs(
    limit: int = 20,
) -> List[Dict[str, Any]]:
    db = SessionLocal()

    try:
        rows = (
            db.query(MESPredictionLog)
            .order_by(MESPredictionLog.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": row.id,
                "production_id": row.production_id,
                "line": row.line,
                "target": row.target,
                "prediction": row.prediction,
                "risk_level": row.risk_level,
                "model_name": row.model_name,
                "algorithm_name": row.algorithm_name,
                "serving_type": row.serving_type,
                "recommendation": row.recommendation,
                "top_influencing_features": row.top_influencing_features,
                "shap_details": row.shap_details,
                "raw_mes_payload": row.raw_mes_payload,
                "mes_timestamp": row.mes_timestamp,
                "created_at": row.created_at,
            }
            for row in rows
        ]

    finally:
        db.close()