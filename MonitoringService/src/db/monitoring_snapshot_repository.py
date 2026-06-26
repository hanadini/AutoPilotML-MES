from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from db.models import MonitoringSnapshot
from schemas.monitoring_snapshot_schema import MonitoringSnapshotResponse


def save_monitoring_snapshot(
    db: Session,
    *,
    total_predictions: int,
    total_quality_results: int,
    coverage_percent: float,
    mae: float,
    mape: float,
    drift_score: float,
    drift_status: str,
    retraining_status: str,
) -> MonitoringSnapshotResponse:
    record = MonitoringSnapshot(
        total_predictions=total_predictions,
        total_quality_results=total_quality_results,
        coverage_percent=coverage_percent,
        mae=mae,
        mape=mape,
        drift_score=drift_score,
        drift_status=drift_status,
        retraining_status=retraining_status,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return MonitoringSnapshotResponse(
        id=record.id,
        snapshot_time=record.snapshot_time,
        total_predictions=record.total_predictions,
        total_quality_results=record.total_quality_results,
        coverage_percent=record.coverage_percent,
        mae=record.mae,
        mape=record.mape,
        drift_score=record.drift_score,
        drift_status=record.drift_status,
        retraining_status=record.retraining_status,
    )


def get_latest_monitoring_snapshots(
    db: Session,
    limit: int = 4,
) -> List[MonitoringSnapshotResponse]:
    rows = (
        db.query(MonitoringSnapshot)
        .order_by(MonitoringSnapshot.snapshot_time.desc())
        .limit(limit)
        .all()
    )

    return [
        MonitoringSnapshotResponse(
            id=row.id,
            snapshot_time=row.snapshot_time,
            total_predictions=row.total_predictions,
            total_quality_results=row.total_quality_results,
            coverage_percent=row.coverage_percent,
            mae=row.mae,
            mape=row.mape,
            drift_score=row.drift_score,
            drift_status=row.drift_status,
            retraining_status=row.retraining_status,
        )
        for row in rows
    ]