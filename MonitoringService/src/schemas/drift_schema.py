from pydantic import BaseModel


class DriftStatusResponse(BaseModel):
    status: str
    drift_score: float

    current_mape: float
    previous_mape: float

    current_samples: int
    previous_samples: int

    message: str