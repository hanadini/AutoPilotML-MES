from __future__ import annotations

import json

from db.database import SessionLocal
from db.mes_prediction_log import MESPredictionLog
from mes.mes_schema import MESPredictionRequest, MESPredictionResponse
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
                request.dict(),
                ensure_ascii=False,
                default=str,
            ),

            mes_timestamp=request.timestamp,
        )

        print("MES INSERT START")
        db.add(log_record)
        db.commit()
        print("MES INSERT SUCCESS")

        logger.info(
            "MES prediction saved | production_id=%s | target=%s | risk=%s",
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