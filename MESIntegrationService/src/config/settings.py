SERVICE_NAME = "MDF1 MESIntegrationService"
API_VERSION = "1.0.0"

MES_API_KEY = "mdf1-mes"

PREDICTION_SERVICE_URL = "http://127.0.0.1:19831"

DATABASE_URL = "postgresql+psycopg2://postgres:1@localhost:5432/mdf1_ml"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# MES_FEATURE_MAPPING_FILE_PATH = METADATA_DIR / "mes_feature_mapping.json"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = BASE_DIR / "artifacts"

MES_FEATURE_MAPPING_FILE_PATH = (
    ARTIFACTS_DIR / "mes_feature_mapping.json"
)

'''
{
  "production_id": "MDF1-2026-001",
  "timestamp": "2026-06-17T09:14:07.249Z",
  "line": "MDF1",
  "target": "labDensityAverage",
  "features": {
    "fibres_density_kg_m3_bulk_density": 730,
    "cooking_time_sec": 180,
    "fiber_density_kg_m3": 145,
    "belt_speed_1_mm_s": 31.2,
    "raw_thickness_mm": 18.2
  }
}'''