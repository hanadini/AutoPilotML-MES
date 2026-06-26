from __future__ import annotations

from typing import Any, Dict, List

from db.database import SessionLocal
from db.mes_prediction_log import MESPredictionLog


def get_latest_mes_prediction_logs(limit: int = 20) -> List[Dict[str, Any]]:
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