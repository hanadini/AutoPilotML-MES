from __future__ import annotations

from typing import Any, Dict, List

from fastapi import HTTPException

from db.mes_log_repository import save_mes_prediction_log
from schemas.mes_schema import MESPredictionRequest, MESPredictionResponse
from services.mes_feature_mapper import map_mes_features_to_ml_features
from services.prediction_client import call_prediction_service
from utils.logging_utils import get_logger


logger = get_logger(__name__)


class MESPredictionService:

    def predict_from_mes(
        self,
        request: MESPredictionRequest,
    ) -> MESPredictionResponse:

        ml_features: Dict[str, Any] = map_mes_features_to_ml_features(
            request.features
        )

        # print("INCOMING MES FEATURES:", request.features)
        # print("MAPPED ML FEATURES:", ml_features)
        # print("MAPPED ML FEATURE COUNT:", len(ml_features))

        logger.info(
            "Incoming MES features | production_id=%s | features=%s",
            request.production_id,
            request.features,
        )

        logger.info(
            "Mapped ML features | production_id=%s | count=%s | features=%s",
            request.production_id,
            len(ml_features),
            ml_features,
        )

        if not ml_features:
            raise HTTPException(
                status_code=400,
                detail="No valid MES features could be mapped to ML features.",
            )

        logger.info(
            "Calling PredictionService | target=%s | feature_count=%s | features=%s",
            request.target,
            len(ml_features),
            ml_features,
        )

        prediction_result = call_prediction_service(
            target=request.target,
            features=ml_features,
        )

        logger.info(
            "PredictionService response | target=%s | response=%s",
            request.target,
            prediction_result,
        )

        recommendations = prediction_result.get("recommendations", [])
        explanation = prediction_result.get("explanation", [])

        recommendation_text = self._build_recommendation_text(
            recommendations
        )

        top_features = self._extract_top_features(
            explanation
        )

        response = MESPredictionResponse(
            production_id=request.production_id,
            target=request.target,
            prediction=float(prediction_result["prediction"]),
            risk_level=str(prediction_result["risk_level"]),
            recommendation=recommendation_text,
            top_influencing_features=top_features,
            model_name=prediction_result.get("model_name"),
            algorithm_name=prediction_result.get("algorithm_name"),
            serving_type=prediction_result.get("serving_type"),
            shap_details=explanation,
        )

        save_mes_prediction_log(
            request=request,
            response=response,
        )

        logger.info(
            "MES prediction completed | production_id=%s | target=%s | risk=%s",
            request.production_id,
            request.target,
            response.risk_level,
        )

        return response

    def _build_recommendation_text(
        self,
        recommendations: List[Dict[str, Any]],
    ) -> str:
        if not recommendations:
            return "Prediction completed. No recommendation was generated."

        messages = [
            str(item.get("message"))
            for item in recommendations
            if item.get("message")
        ]

        if not messages:
            return "Prediction completed. No recommendation message was available."

        return " ".join(messages)

    def _extract_top_features(
        self,
        explanation: List[Dict[str, Any]],
    ) -> List[str]:
        return [
            str(item.get("feature"))
            for item in explanation
            if item.get("feature")
        ][:5]