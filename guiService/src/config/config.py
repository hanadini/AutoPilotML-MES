from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# PredictionService
# ============================================================

PREDICTION_SERVICE_HOST = "127.0.0.1"
PREDICTION_SERVICE_PORT = 19831

PREDICTION_SERVICE_BASE_URL = (
    f"http://{PREDICTION_SERVICE_HOST}:{PREDICTION_SERVICE_PORT}"
)

PREDICT_ALL_ENDPOINT = (
    f"{PREDICTION_SERVICE_BASE_URL}/predict-all"
)

EXPLAIN_ENDPOINT = (
    f"{PREDICTION_SERVICE_BASE_URL}/explain"
)

HEALTH_ENDPOINT = (
    f"{PREDICTION_SERVICE_BASE_URL}/health"
)

MODELS_ENDPOINT = (
    f"{PREDICTION_SERVICE_BASE_URL}/models"
)

# ============================================================
# MonitoringService
# ============================================================

MONITORING_SERVICE_HOST = "127.0.0.1"
MONITORING_SERVICE_PORT = 19834

MONITORING_SERVICE_BASE_URL = (
    f"http://{MONITORING_SERVICE_HOST}:{MONITORING_SERVICE_PORT}"
)

MONITORING_HEALTH_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/health"
)

SYSTEM_OVERVIEW_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/system-overview"
)

PREDICTION_STATS_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/prediction-stats"
)

FEEDBACK_STATS_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/feedback-stats"
)

MODEL_PERFORMANCE_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/model-performance"
)

MODEL_PERFORMANCE_BY_TARGET_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/model-performance-by-target"
)

RETRAINING_ADVISOR_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/retraining-advisor"
)

DRIFT_STATUS_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/drift-status"
)

MONITORING_DASHBOARD_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/dashboard"
)

MONITORING_SNAPSHOTS_ENDPOINT = (
    f"{MONITORING_SERVICE_BASE_URL}/api/v1/monitoring/snapshots"
)

REQUEST_TIMEOUT_SECONDS = 20

# ============================================================
# Runtime folders
# ============================================================

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "prediction_logs.json"

# ============================================================
# MDF1 targets
# ============================================================

ALL_TARGETS = [
    "labDensityAverage",
    "labBendingAvg",
    "labEModulAvg",
    "labTensileAvg",
    "labSurfaceSoundnessAvg",
]

TARGET_LABELS = {
    "labDensityAverage": "Density",
    "labBendingAvg": "Bending Strength",
    "labEModulAvg": "E-Modulus",
    "labTensileAvg": "Tensile Strength",
    "labSurfaceSoundnessAvg": "Surface Soundness",
}

TARGET_UNITS = {
    "labDensityAverage": "kg/m³",
    "labBendingAvg": "N/mm²",
    "labEModulAvg": "N/mm²",
    "labTensileAvg": "N/mm²",
    "labSurfaceSoundnessAvg": "N/mm²",
}

TARGET_THRESHOLDS = {
    "labDensityAverage": {
        "danger_low": 720,
        "warning_low": 730,
        "warning_high": 750,
        "danger_high": 760,
        "higher_is_better": False,
    },
    "labBendingAvg": {
        "danger_low": 30,
        "warning_low": 31,
        "warning_high": None,
        "danger_high": None,
        "higher_is_better": True,
    },
    "labEModulAvg": {
        "danger_low": 2800,
        "warning_low": 3000,
        "warning_high": None,
        "danger_high": None,
        "higher_is_better": True,
    },
    "labTensileAvg": {
        "danger_low": 0.35,
        "warning_low": 0.40,
        "warning_high": None,
        "danger_high": None,
        "higher_is_better": True,
    },
    "labSurfaceSoundnessAvg": {
        "danger_low": 0.80,
        "warning_low": 0.90,
        "warning_high": None,
        "danger_high": None,
        "higher_is_better": True,
    },
}

# ============================================================
# GUI process features
# ============================================================

CORE_PROCESS_FEATURES = [
    "rawThickness",
    "beltSpeed1",
    "pressPressureGlobal_mean",
    "pressPressureMid_mean",
]

FORCED_PRODUCTION_FEATURES = [
    "operCookingTime",
    "operFibresDensity",
    "operPressFibreDensity",
]

DEFAULT_MANUAL_FEATURE_VALUES = {
    "rawThickness": 18.2,
    "beltSpeed1": 31.2,
    "pressPressureGlobal_mean": 142.5,
    "pressPressureMid_mean": 142.5,
    "operCookingTime": 180,
    "operFibresDensity": 145,
    "operPressFibreDensity": 720,
}

CONTROL_FEATURE_LABELS = {
    "rawThickness": "Raw Thickness",
    "beltSpeed1": "Belt Speed 1",
    "pressPressureGlobal_mean": "Global Press Pressure",
    "pressPressureMid_mean": "Mid Press Pressure",
    "operCookingTime": "Cooking Time",
    "operFibresDensity": "Fibres Density",
    "operPressFibreDensity": "Press Fibre Density",
}

CONTROL_FEATURE_UNITS = {
    "rawThickness": "mm",
    "beltSpeed1": "m/min",
    "pressPressureGlobal_mean": "bar",
    "pressPressureMid_mean": "bar",
    "operCookingTime": "sec",
    "operFibresDensity": "kg/m³",
    "operPressFibreDensity": "kg/m³",
}

# ============================================================
# GUI options
# ============================================================

GUI_REFRESH_SECONDS = 10

HISTORY_DEFAULT_LIMIT = 20
HISTORY_ALLOWED_LIMITS = [10, 20, 50]

# ============================================================
# Risk / styling constants
# ============================================================

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"

ROW_CLASS_NORMAL = ""
ROW_CLASS_WARNING = "row-warning"
ROW_CLASS_ALERT = "row-alert"

PREDICTION_CLASS_NORMAL = "prediction-normal"
PREDICTION_CLASS_WARNING = "prediction-warning"
PREDICTION_CLASS_HIGH = "prediction-high"

ALIASES = {
    "GREEN": RISK_LOW,
    "LOW": RISK_LOW,
    "YELLOW": RISK_MEDIUM,
    "MEDIUM": RISK_MEDIUM,
    "RED": RISK_HIGH,
    "HIGH": RISK_HIGH,
}

# ============================================================
# Runtime setup
# ============================================================


def ensure_runtime_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


ensure_runtime_directories()