from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class FeatureImpact(BaseModel):
    feature: str = Field(
        ...,
        description="Feature name contributing to the prediction.",
    )

    value: Any = Field(
        default=None,
        description="Input value used for this feature.",
    )

    impact: float = Field(
        ...,
        description=(
            "SHAP impact value. Positive values increase prediction, "
            "negative values decrease prediction."
        ),
    )

    absolute_impact: float = Field(
        ...,
        description="Absolute SHAP impact magnitude.",
    )

    direction: str = Field(
        ...,
        description="Impact direction: increase or decrease.",
    )

    is_watched_feature: bool = Field(
        default=False,
        description="Whether the feature is marked as a watched MDF1 feature.",
    )


class RecommendationItem(BaseModel):
    feature: str = Field(
        ...,
        description="Original feature name.",
    )

    feature_label: str = Field(
        ...,
        description="Human-friendly feature label.",
    )

    impact: float = Field(
        ...,
        description="SHAP impact value for the recommendation.",
    )

    direction: str = Field(
        ...,
        description="Whether the feature increases or decreases the prediction.",
    )

    risk_level: str = Field(
        ...,
        description="Associated risk level.",
    )

    message: str = Field(
        ...,
        description="Operator-oriented recommendation message.",
    )

    is_watched_feature: bool = Field(
        default=False,
        description="Whether this recommendation is based on a watched feature.",
    )


class PredictionResponse(BaseModel):
    target: str = Field(
        ...,
        description=(
            "Requested MDF1 target, for example labDensityAverage "
            "or labBendingAvg."
        ),
    )

    model_name: str = Field(
        ...,
        description="Resolved artifact model name.",
    )

    algorithm_name: str = Field(
        ...,
        description=(
            "Resolved algorithm or serving strategy "
            "(xgb, rf, weighted_ensemble, etc.)."
        ),
    )

    prediction: float = Field(
        ...,
        description="Predicted target value.",
    )

    used_feature_count: int = Field(
        ...,
        description="Number of features used during prediction.",
    )

    missing_features: List[str] = Field(
        default_factory=list,
        description=(
            "Missing features from the request payload."
        ),
    )

    extra_features: List[str] = Field(
        default_factory=list,
        description=(
            "Extra incoming features not expected by features.json."
        ),
    )

    invalid_numeric_features: List[str] = Field(
        default_factory=list,
        description=(
            "Features containing invalid numeric values."
        ),
    )

    null_features: List[str] = Field(
        default_factory=list,
        description=(
            "Features containing null/NaN values."
        ),
    )

    forced_missing_features: List[str] = Field(
        default_factory=list,
        description=(
            "Required MDF1 production features missing from input."
        ),
    )

    risk_level: str = Field(
        default="LOW",
        description="Calculated risk level.",
    )

    serving_type: str = Field(
        default="single_model",
        description="Serving strategy used for prediction.",
    )

    explanation_model_name: str = Field(
        default="",
        description="Artifact model used for SHAP explanation.",
    )

    explanation_algorithm: str = Field(
        default="",
        description="Algorithm used for SHAP explanation.",
    )

    explanation: List[FeatureImpact] = Field(
        default_factory=list,
        description="Top SHAP feature impacts.",
    )

    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Operator-oriented recommendations.",
    )

    duration_ms: float = Field(
        default=0.0,
        description="Prediction processing duration in milliseconds.",
    )

    ensemble_members: Dict[str, str] = Field(
        default_factory=dict,
        description="Ensemble member model mapping.",
    )

    ensemble_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Ensemble member weights.",
    )

    member_predictions: Dict[str, float] = Field(
        default_factory=dict,
        description="Prediction value from each ensemble member.",
    )


class PredictAllItemResponse(BaseModel):
    model_name: str = Field(
        ...,
        description="Resolved artifact model name.",
    )

    algorithm_name: str = Field(
        ...,
        description="Resolved algorithm or serving strategy.",
    )

    prediction: float = Field(
        ...,
        description="Predicted target value.",
    )

    risk_level: str = Field(
        ...,
        description="Calculated risk level.",
    )

    explanation: List[FeatureImpact] = Field(
        default_factory=list,
        description="Top SHAP feature impacts.",
    )

    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Operator-oriented recommendations.",
    )


class PredictAllResponse(BaseModel):
    predictions: Dict[str, PredictAllItemResponse] = Field(
        ...,
        description="Predictions for all registered MDF1 targets.",
    )

    used_feature_count_by_target: Dict[str, int] = Field(
        ...,
        description="Feature count used per target.",
    )

    important_features_by_target: Dict[str, List[str]] = Field(
        ...,
        description="Important features per target.",
    )