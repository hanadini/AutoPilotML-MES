from __future__ import annotations

from pathlib import Path

# ============================================================
# Base paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARIES_DIR = REPORTS_DIR / "summaries"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FEATURE_LISTS_DIR = ARTIFACTS_DIR / "feature_lists"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "training.log"

RETENTION_REPORT_PATH = TABLES_DIR / "data_retention_report.csv"
TARGET_AVAILABILITY_REPORT_PATH = TABLES_DIR / "target_availability_report.csv"
TARGET_COMBINATION_REPORT_PATH = TABLES_DIR / "target_combination_report.csv"
RETENTION_SUMMARY_PATH = SUMMARIES_DIR / "data_retention_summary.txt"

# ============================================================
# Raw data settings
# ============================================================

RAW_FILE_PATTERN = "*.xlsx"
SHEET_NAME = "FQTTS_EXCEL_EXPORT"
HEADER_ROW_FOR_TECHNICAL_NAMES = 1

# ============================================================
# Data outputs
# ============================================================

FLATTENED_FILE_PATH = INTERIM_DIR / "mdf1_flattened_data.csv"
CLEANED_FILE_PATH = INTERIM_DIR / "mdf1_cleaned_data.csv"

COLUMN_MAPPING_FILE_PATH = METADATA_DIR / "column_mappings.csv"
MES_FEATURE_MAPPING_FILE_PATH = METADATA_DIR / "mes_feature_mapping.json"
VALIDATION_REPORT_PATH = METADATA_DIR / "validation_report.json"
AUDIT_SUMMARY_PATH = SUMMARIES_DIR / "initial_data_audit.txt"

TIMING_REPORT_PATH = TABLES_DIR / "training_timing_report.csv"

# ============================================================
# Data cleaning settings
# ============================================================

TEXT_NA_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "None",
    "-",
    "--",
    "null",
    "NULL",
}

MAX_MISSING_RATIO_PER_FEATURE = 0.40
DROP_ROWS_WITH_MISSING_PRIMARY_TARGETS = True
DEFAULT_IMPUTER_STRATEGY = "median"

# ============================================================
# General training settings
# ============================================================

RANDOM_STATE = 42
IID_TEST_SIZE = 0.20
IID_VALID_SIZE = 0.20

N_JOBS = -1
VERBOSE = -1

# ============================================================
# Evaluation benchmark settings
# ============================================================

ENABLE_RANDOM_SPLIT_BENCHMARK = True
ENABLE_TIME_SPLIT_BENCHMARK = True
ENABLE_TIME_CV_BENCHMARK = True

# ============================================================
# Random Forest settings
# ============================================================

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 12

# ============================================================
# XGBoost tuning settings
# ============================================================

XGB_TUNING_ITERATIONS = 15
XGB_EARLY_STOPPING_ROUNDS = 30

XGB_PARAM_DISTRIBUTION = {
    "n_estimators": [400, 600, 800, 1000, 1200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.08],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 9],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
    "reg_alpha": [0.0, 0.1, 0.3, 1.0, 3.0],
    "reg_lambda": [1.0, 3.0, 5.0, 10.0, 15.0],
}

# ============================================================
# LightGBM settings
# ============================================================

LGBM_N_ESTIMATORS = 400
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 31
LGBM_SUBSAMPLE = 0.8
LGBM_COLSAMPLE_BYTREE = 0.8

# ============================================================
# Ensemble settings
# ============================================================

ENSEMBLE_WEIGHTS = [0.55, 0.60, 0.65, 0.70, 0.75]
ENSEMBLE_OVERFIT_PENALTY = 0.20
ENSEMBLE_MIN_GAIN = 0.001

# ============================================================
# Time-based cross-validation settings
# ============================================================

TIME_CV_N_FOLDS = 3
TIME_CV_MIN_TRAIN_RATIO = 0.50
TIME_CV_VAL_RATIO = 0.15

# ============================================================
# Artifact governance
# ============================================================

BEST_MODELS_REGISTRY_PATH = ARTIFACTS_DIR / "best_models_registry.json"
ARTIFACTS_ROOT = "artifacts"

KEEP_BEST_MODELS_REGISTRY = True
ENABLE_ARTIFACT_CLEAN_BEFORE_RUN = True

# ============================================================
# Report directories
# ============================================================

COMPARISON_REPORTS_DIR = REPORTS_DIR / "comparisons"
EVALUATION_COMPARISON_DIR = REPORTS_DIR / "evaluation_comparison"

SHAP_EXPLAINABILITY_DIR = REPORTS_DIR / "shap_explainability"
PROCESS_EXPLAINABILITY_DIR = REPORTS_DIR / "process_explainability"
ML_PROCESS_IMPORTANCE_DIR = REPORTS_DIR / "ml_process_importance"
FINAL_EXPLAINABILITY_DIR = REPORTS_DIR / "final_explainability"
PROCESS_RECOMMENDATION_DIR = REPORTS_DIR / "process_recommendations"
PROCESS_GROUP_SUMMARY_DIR = REPORTS_DIR / "process_group_summary"

# ============================================================
# Explainability settings
# ============================================================

SHAP_MAX_ROWS = 500

FINAL_EXPLAINABILITY_FILE_NAME = "all_targets_final_explainability.csv"
ML_PROCESS_IMPORTANCE_FILE_NAME = "all_targets_ml_process_importance.csv"
SHAP_IMPORTANCE_FILE_NAME = "all_targets_shap_importance.csv"

PROCESS_RECOMMENDATION_TOP_N = 10

# ============================================================
# Data quality / process impact settings
# ============================================================

DATA_QUALITY_HIGH_MISSING_THRESHOLD = 0.40
DATA_QUALITY_MODERATE_MISSING_THRESHOLD = 0.20

PROCESS_IMPACT_REQUIRED_MIN_VALID_PAIRS = 3

PROCESS_IMPACT_NOTE = (
    "Correlation shows statistical association, "
    "not guaranteed causality."
)

# ============================================================
# Logging style
# ============================================================

TRAINING_LOG_SEPARATOR = "=" * 70

# ============================================================
# Directory creation
# ============================================================

DIRECTORIES_TO_CREATE = [
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    METADATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    SUMMARIES_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FEATURE_LISTS_DIR,
    LOG_DIR,
    COMPARISON_REPORTS_DIR,
    EVALUATION_COMPARISON_DIR,
    SHAP_EXPLAINABILITY_DIR,
    PROCESS_EXPLAINABILITY_DIR,
    ML_PROCESS_IMPORTANCE_DIR,
    FINAL_EXPLAINABILITY_DIR,
    PROCESS_RECOMMENDATION_DIR,
    PROCESS_GROUP_SUMMARY_DIR,
]

for directory in DIRECTORIES_TO_CREATE:
    directory.mkdir(parents=True, exist_ok=True)