from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

# ============================================================
# Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = ARTIFACTS_DIR / "best_models_registry.json"

LOG_FILE = LOG_DIR / "prediction-service.log"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================
# Artifact synchronization
# ============================================================

TRAINING_SERVICE_DIR = BASE_DIR.parent / "TrainingService"

TRAINING_ARTIFACTS_DIR = TRAINING_SERVICE_DIR / "artifacts"
PREDICTION_ARTIFACTS_DIR = ARTIFACTS_DIR

# ============================================================
# API settings
# ============================================================

SERVICE_NAME = "MDF1 PredictionService"
API_VERSION = "1.0.0"

DEFAULT_HOST = os.getenv("PREDICTION_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PREDICTION_PORT", "19831"))

MES_API_KEY = "mdf1-mes-secret-key"
# ============================================================
# Default model / target mapping
# ============================================================

DEFAULT_TARGET = os.getenv("DEFAULT_TARGET", "labDensityAverage")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "labDensityAverage_rf_v1")

TARGET_MODEL_MAP = {
    "labDensityAverage": "labDensityAverage_rf_v1",
    "labBendingAvg": "labBendingAvg_rf_v1",
    "labEModulAvg": "labEModulAvg_rf_v1",
    "labTensileAvg": "labTensileAvg_rf_v1",
    "labSurfaceSoundnessAvg": "labSurfaceSoundnessAvg_rf_v1",
}

SUPPORTED_TARGETS = list(TARGET_MODEL_MAP.keys())

# ============================================================
# Artifact bundle contract
# ============================================================

MODEL_FILE_NAME = "model.joblib"
FEATURES_FILE_NAME = "features.json"
METRICS_FILE_NAME = "metrics.json"
METADATA_FILE_NAME = "metadata.json"

# ============================================================
# TrainingService alignment
# ============================================================

TRAINING_SERVICE_CONTRACT_VERSION = "1.0.0"

TRAINING_EVALUATION_PHILOSOPHY = (
    "time_based_industrial_validation"
)

REQUIRE_FEATURES_JSON = True
REQUIRE_METADATA_JSON = True
REQUIRE_METRICS_JSON = True
REQUIRE_MODEL_JOBLIB = True

# ============================================================
# Dynamic required artifact contract
# ============================================================
REQUIRED_ARTIFACT_FILES = []

if REQUIRE_MODEL_JOBLIB:
    REQUIRED_ARTIFACT_FILES.append(
        MODEL_FILE_NAME
    )

if REQUIRE_FEATURES_JSON:
    REQUIRED_ARTIFACT_FILES.append(
        FEATURES_FILE_NAME
    )

if REQUIRE_METRICS_JSON:
    REQUIRED_ARTIFACT_FILES.append(
        METRICS_FILE_NAME
    )

if REQUIRE_METADATA_JSON:
    REQUIRED_ARTIFACT_FILES.append(
        METADATA_FILE_NAME
    )
# ============================================================
# Feature governance
# ============================================================

STRICT_FEATURE_ORDER = True
ALLOW_EXTRA_FEATURES = True
ALLOW_MISSING_FEATURES = False

ENABLE_TYPE_COERCION = True
ENABLE_NAN_GUARD = True
ENABLE_INPUT_RANGE_CHECKS = False

# ============================================================
# Ensemble serving
# ============================================================

ENABLE_WEIGHTED_ENSEMBLES = True
DEFAULT_REGISTRY_MODE = "single_model"

# ============================================================
# Explainability
# ============================================================

ENABLE_EXPLAINABILITY = True
ENABLE_SHAP_OUTPUTS = True
MAX_EXPLAINED_FEATURES = 15

SHAP_FILE_NAMES = [
    "shap_summary.png",
    "shap_values.joblib",
    "shap_feature_importance.csv",
]

# ============================================================
# MDF1 production-safe logic
# ============================================================

ENABLE_FORCED_PRODUCTION_FEATURE_CHECK = True
ENABLE_INDUSTRIAL_SAFE_PREPROCESSING = True

FORCED_PRODUCTION_FEATURES_BY_TARGET  = {}

# ============================================================
# Database / logging
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:1@localhost:5432/mdf1_ml",
)

ENABLE_DB_LOGGING = True
ENABLE_JSON_LOGGING = True

PREDICTION_LOG_FILE = LOG_DIR / "prediction_logs.json"

# ============================================================
# Feature labels for explanations and monitoring
# ============================================================

FEATURE_LABELS = {
    "pressPressureMid_mean": "mid press pressure",
    "pressPressureGlobal_mean": "global press pressure",
    "operCookingTime": "cooking time",
    "operFibresDensity": "fibres density",
    "operPressFibreDensity": "press fibre density",
    "pressPressure": "press pressure",
    "beltSpeed": "belt speed",
    "rawThickness": "raw board thickness",
    "thicknessClosed": "closed thickness",
    "temp": "temperature",
}


WATCH_FEATURES = {
    "pressPressureMid_mean",
    "pressPressureGlobal_mean",
    "operCookingTime",
    "operFibresDensity",
    "operPressFibreDensity",
}
# ============================================================
# Risk rules for monitoring
# ============================================================

RISK_RULES: Dict[str, Dict[str, Any]] = {
    "labDensityAverage": {
        "type": "range",
        "low": 0,
        "medium": 700,
        "high": 800,
        "direction": "high_is_risky",
    },
    "labBendingAvg": {
        "type": "range",
        "low": 0,
        "medium": 35,
        "high": 30,
        "direction": "low_is_risky",
    },
    "labEModulAvg": {
        "type": "range",
        "low": 0,
        "medium": 2500,
        "high": 2200,
        "direction": "low_is_risky",
    },
    "labTensileAvg": {
        "type": "range",
        "low": 0,
        "medium": 0.45,
        "high": 0.35,
        "direction": "low_is_risky",
    },
    "labSurfaceSoundnessAvg": {
        "type": "range",
        "low": 0,
        "medium": 1.10,
        "high": 0.90,
        "direction": "low_is_risky",
    },
}

# ============================================================
# Default feature example for documentation and testing
# ============================================================
DEFAULT_FEATURE_EXAMPLE: Dict[str, Any] = {
    "rawThickness": 18.2,
    "beltSpeed1": 31.2,
    "pressPressureMid_mean": 142.5,
    "pressPressureGlobal_mean": 142.5,
    "operCookingTime": 180,
    "operFibresDensity": 145,
    "operPressFibreDensity": 720,
}