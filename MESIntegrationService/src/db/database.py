from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import DATABASE_URL
from utils.logging_utils import get_logger


logger = get_logger(__name__)


engine = create_engine(
    DATABASE_URL,
    echo=False, #sql queries are hidden->clean logs True -> sql queries are shown in logs and useful for debugging
    future=True, # use SQLAlchemy 2.0 style, even in 1.4
    pool_pre_ping=True, #Before using a database connection,check if it's still alive.
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()

logger.info("MESIntegrationService database engine initialized.")