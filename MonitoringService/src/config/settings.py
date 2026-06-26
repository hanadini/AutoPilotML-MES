from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

SERVICE_NAME = "MDF1 MonitoringService"
API_VERSION = "1.0.0"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19834

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:1@localhost:5432/mdf1_ml",
)

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


'''
Retraining Advisor Statuses:
NOT_ENOUGH_DATA
WATCH
RETRAINING_RECOMMENDED
HEALTHY
'''