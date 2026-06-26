from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

SERVICE_NAME = "MDF1 OperatorFeedbackService"
API_VERSION = "1.0.0"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19833

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:1@localhost:5432/mdf1_ml",
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


VALID_DECISIONS = {
    "ACCEPTED",
    "REJECTED",
    "IGNORED",
    "MODIFIED",
}

'''
{
  "production_id": "MDF1-MES-TEST-999",
  "target": "labDensityAverage",
  "prediction_value": "712.4",
  "risk_level": "MEDIUM",
  "recommendation": "Review belt speed and press fibre density.",
  "operator_decision": "ACCEPTED",
  "operator_comment": "Applied recommendation during shift B.",
  "operator_name": "Operator A"
}'''