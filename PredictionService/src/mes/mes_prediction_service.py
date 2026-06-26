from __future__ import annotations

from http.client import responses
from typing import Any, Dict, List

from fastapi import HTTPException

from config.settings import SUPPORTED_TARGETS
from mes.mes_schema import MESPredictionRequest, MESPredictionResponse
from mes.mes_feature_mapper import map_mes_features_to_ml_features
from mes.mes_prediction_logger import save_mes_prediction_log
from service.explanation_service import explain_prediction


class MESPredictionService:

    def predict_from_mes(
        self,
        request: MESPredictionRequest,
    ) -> MESPredictionResponse:

        ml_features: Dict[str, Any] = map_mes_features_to_ml_features(
            request.features
        )

        if not ml_features:
            raise HTTPException(
                status_code=400,
                detail="No valid MES features could be mapped to ML features.",
            )

        if request.target not in SUPPORTED_TARGETS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target '{request.target}'. Supported targets: {SUPPORTED_TARGETS}",
            )

        try:
            prediction_result = explain_prediction(
                target=request.target,
                incoming_features=ml_features,
                top_n=5,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"MES prediction failed: {str(exc)}",
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