from __future__ import annotations

from fastapi import APIRouter, Depends

from db.mes_log_repository import get_latest_mes_prediction_logs
from schemas.mes_schema import MESPredictionRequest, MESPredictionResponse
from security.mes_security import verify_mes_api_key
from services.mes_prediction_service import MESPredictionService

from services.prediction_client import check_prediction_service_health


router = APIRouter(
    prefix="/api/v1/mes",
    tags=["MES Integration"],
)

mes_prediction_service = MESPredictionService()


@router.post(
    "/predict",
    response_model=MESPredictionResponse,
    dependencies=[Depends(verify_mes_api_key)],
)


def predict_mes(
    request: MESPredictionRequest,
) -> MESPredictionResponse:
    return mes_prediction_service.predict_from_mes(request)


@router.get(
    "/logs",
    dependencies=[Depends(verify_mes_api_key)],
)
def get_mes_logs(
    limit: int = 20,
):
    return get_latest_mes_prediction_logs(limit=limit)


@router.get(
    "/prediction-service-health",
    dependencies=[Depends(verify_mes_api_key)],
)
def prediction_service_health():
    return check_prediction_service_health()



'''
{
  "production_id": "MDF1-2026-001",
  "timestamp": "2026-06-15T12:00:00",
  "line": "MDF1",
  "target": "labDensityAverage",
  "features": {
    "press_fibre_density": 720,
    "cooking_time": 180,
    "fibres_density": 145,
    "press_pressure_mid": 142.5,
    "press_pressure_global": 142.5,
    "belt_speed": 31.2,
    "raw_thickness": 18.2
  }
}
'''