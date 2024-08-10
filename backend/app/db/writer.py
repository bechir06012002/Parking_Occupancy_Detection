"""Async DB writer for occupancy_state/occupancy_events/pipeline_runs.

occupancy_state is upserted, occupancy_events is append-only —
on a transition we upsert every column and add an event row; on a
no-change tick we only touch last_seen_at. `transitioned` is decided by the
caller (services/pipeline/run_pipeline.py), which knows whether this tick's
call to OccupancySmoother.observe() actually flipped the spot — see there
for why every spot's very first observation is guaranteed to count as a
transition, so occupancy_state always has a row before a non-transition
update ever targets it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import OccupancyEvent, OccupancyState, PipelineRun
from app.domain.occupancy import SpotState
from app.domain.pipeline import PipelineRunSummary


async def record_spot_observation(
    session: AsyncSession, state: SpotState, *, transitioned: bool, now: datetime
) -> None:
    if not transitioned:
        await session.execute(
            update(OccupancyState)
            .where(OccupancyState.spot_id == state.spot_id)
            .values(last_seen_at=now)
        )
        return

    stmt = pg_insert(OccupancyState).values(
        spot_id=state.spot_id,
        is_occupied=state.is_occupied,
        confidence=state.confidence,
        track_id=state.track_id,
        last_changed_at=now,
        last_seen_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[OccupancyState.spot_id],
        set_={
            "is_occupied": stmt.excluded.is_occupied,
            "confidence": stmt.excluded.confidence,
            "track_id": stmt.excluded.track_id,
            "last_changed_at": stmt.excluded.last_changed_at,
            "last_seen_at": stmt.excluded.last_seen_at,
        },
    )
    await session.execute(stmt)
    session.add(
        OccupancyEvent(
            spot_id=state.spot_id,
            is_occupied=state.is_occupied,
            confidence=state.confidence,
            occurred_at=now,
        )
    )


async def record_pipeline_run(session: AsyncSession, summary: PipelineRunSummary) -> None:
    session.add(
        PipelineRun(
            camera_id=summary.camera_id,
            started_at=summary.started_at,
            ended_at=summary.ended_at,
            frames_processed=summary.frames_processed,
            avg_latency_ms=summary.avg_latency_ms,
            p95_latency_ms=summary.p95_latency_ms,
            model_version=summary.model_version,
        )
    )
