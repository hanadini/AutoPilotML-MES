from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from config.settings import DEFAULT_FEATURE_EXAMPLE


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target": "labDensityAverage",
                "features": DEFAULT_FEATURE_EXAMPLE,
            }
        }
    )

    target: str = Field(
        ...,
        description=(
            "Registered MDF1 target name, for example: "
            "labDensityAverage, labBendingAvg, labEModulAvg, "
            "labTensileAvg, or labSurfaceSoundnessAvg."
        ),
        example="labDensityAverage",
    )

    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Incoming feature dictionary. Keys must match the saved "
            "features.json contract for the selected target/model."
        ),
        example=DEFAULT_FEATURE_EXAMPLE,
    )


class PredictAllRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "production_order": "TEST-001",
                "board_id": "BOARD-TEST-001",
                "request_source": "Swagger-Test",
                "features": DEFAULT_FEATURE_EXAMPLE,
            }
        }
    )

    production_order: Optional[str] = Field(
        default=None,
        description="Optional production order reference.",
        example="TEST-001",
    )

    board_id: Optional[str] = Field(
        default=None,
        description="Optional board identifier.",
        example="BOARD-TEST-001",
    )

    request_source: str = Field(
        default="PredictionService",
        description=(
            "Source of the prediction request, for example Swagger, GUI, MES, "
            "or scheduled integration."
        ),
        example="Swagger-Test",
    )

    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Incoming feature dictionary used for all registered target "
            "predictions. Each target will enforce its own saved features.json."
        ),
        example=DEFAULT_FEATURE_EXAMPLE,
    )