from fastapi import APIRouter, Depends

from mes.mes_log_service import get_latest_mes_prediction_logs
from mes.mes_prediction_service import MESPredictionService
from mes.mes_schema import MESPredictionRequest, MESPredictionResponse
from mes.mes_security import verify_mes_api_key


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
def predict_mes(request: MESPredictionRequest) -> MESPredictionResponse:
    return mes_prediction_service.predict_from_mes(request)


@router.get(
    "/logs",
    dependencies=[Depends(verify_mes_api_key)],
)
def get_mes_logs(limit: int = 20):
    return get_latest_mes_prediction_logs(limit=limit)