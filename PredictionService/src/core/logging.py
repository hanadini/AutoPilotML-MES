from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timedelta
from logging.handlers import BaseRotatingHandler
from pathlib import Path
from typing import Optional


class DailySizeRotatingHandler(BaseRotatingHandler):
    """
    Rotates the active log file:
    - when the date changes
    - when file size exceeds max_bytes

    Archived files are stored as:
        <stem>-YYYY-MM-DD.<index>.log
    """

    def __init__(
        self,
        filename: Path | str,
        *,
        archived_dir: Path | str | None = None,
        max_bytes: int = 50 * 1024 * 1024,
        retention_days: int = 30,
        encoding: str = "utf-8",
    ) -> None:
        self.base_path = Path(filename)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(str(self.base_path), mode="a", encoding=encoding, delay=True)

        if not self.base_path.exists():
            self.base_path.touch()

        self.archived_dir = Path(archived_dir) if archived_dir else self.base_path.parent / "archived"
        self.archived_dir.mkdir(parents=True, exist_ok=True)

        self.log_file_stem = self.base_path.stem
        self.max_bytes = max_bytes
        self.retention_days = retention_days

    @property
    def current_date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    @property
    def log_file_mtime_date(self) -> str:
        if not self.base_path.exists():
            return self.current_date
        return datetime.fromtimestamp(self.base_path.stat().st_mtime).strftime("%Y-%m-%d")

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802
        if not self.base_path.exists():
            return False

        file_too_old = self.log_file_mtime_date != self.current_date
        file_too_large = self.base_path.stat().st_size >= self.max_bytes
        return file_too_old or file_too_large

    def doRollover(self) -> None:  # noqa: N802
        if self.stream:
            self.stream.close()
            self.stream = None

        rollover_date = self.log_file_mtime_date
        next_index = self._get_next_chunk_index(rollover_date)

        archived_name = self.archived_dir / f"{self.log_file_stem}-{rollover_date}.{next_index}.log"
        if self.base_path.exists():
            self.base_path.rename(archived_name)

        self._cleanup_old_logs()
        self.stream = self._open()

    def _get_next_chunk_index(self, date_str: str) -> int:
        pattern = re.compile(rf"{re.escape(self.log_file_stem)}-{date_str}\.(\d+)\.log$")
        indices: list[int] = []

        for log_file in self.archived_dir.glob(f"{self.log_file_stem}-{date_str}.*.log"):
            match = pattern.match(log_file.name)
            if match:
                indices.append(int(match.group(1)))

        return max(indices, default=-1) + 1

    def _cleanup_old_logs(self) -> None:
        cutoff = datetime.now() - timedelta(days=self.retention_days)

        for log_file in self.archived_dir.glob(f"{self.log_file_stem}-*.log"):
            try:
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", log_file.name)
                if not date_match:
                    continue
                log_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                if log_date < cutoff:
                    log_file.unlink(missing_ok=True)
            except Exception:
                continue


def configure_logging(
    *,
    service_name: str,
    log_dir: Path | str = "logs",
    log_level: Optional[str] = None,
    max_mb: int = 50,
    retention_days: int = 30,
    log_to_console: bool = True,
) -> None:
    """
    Configure root logging once per service process.
    """
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [
        DailySizeRotatingHandler(
            filename=log_dir / f"{service_name}.log",
            max_bytes=max_mb * 1024 * 1024,
            retention_days=retention_days,
        )
    ]

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(console_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)