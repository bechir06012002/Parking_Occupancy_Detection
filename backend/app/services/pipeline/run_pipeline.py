"""The pipeline Facade: capture -> detect -> track -> map -> smooth -> persist.

`run_pipeline` is the worker's one entrypoint (`worker/run_worker.py`) into
capture, detection, tracking, and occupancy mapping. It stays fully
decoupled from SQLAlchemy — persistence and per-run bookkeeping are
injected async callbacks, not a DB session — so it can be unit-tested with
fakes like every other service here, with no real DB, cv2, or torch
involved.

Batch scheduling: each tick, every camera's `FrameGrabber.maybe_grab` is
polled; cameras that aren't due yet contribute nothing. Every frame that
*is* ready across every camera goes into a single `detector.predict_batch`
call — batching amortizes inference overhead as camera count grows — not
one call per camera.

Latency: for each frame, elapsed time from the moment it was captured to
the moment every one of its spots has been persisted is recorded; avg/p95
across the run feed the `PipelineRunSummary` handed to `on_run_finished`
once per camera, in a `finally` block so an interrupted (Ctrl+C/cancelled)
continuous run still gets a row instead of losing the run's stats.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from app.domain.detection import Detection
from app.domain.occupancy import Spot, SpotState
from app.domain.pipeline import PipelineRunSummary
from app.domain.tracking import Track
from app.services.frame import Frame
from app.services.occupancy.matching import OccupancyStrategy
from app.services.occupancy.state import OccupancySmoother

logger = logging.getLogger(__name__)


class _GrabberLike(Protocol):
    def maybe_grab(self, *, now: float | None = None) -> Frame | None: ...


class _TrackerLike(Protocol):
    def update(self, detections: list[Detection]) -> list[Track]: ...


class _DetectorLike(Protocol):
    def predict_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]: ...


@dataclass
class CameraRuntime:
    camera_id: int
    spots: list[Spot]
    grabber: _GrabberLike
    tracker: _TrackerLike


PersistSpotState = Callable[[SpotState, bool], Awaitable[None]]
OnRunFinished = Callable[[PipelineRunSummary], Awaitable[None]]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


@dataclass
class _RunStats:
    frames_processed: int = 0
    latencies_ms: list[float] = field(default_factory=list)


async def run_pipeline(
    *,
    cameras: Sequence[CameraRuntime],
    detector: _DetectorLike,
    strategy: OccupancyStrategy,
    smoothing_window: int,
    persist: PersistSpotState,
    model_version: str,
    tick_interval_seconds: float,
    on_run_finished: OnRunFinished | None = None,
    max_ticks: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    started_at = now_factory()
    stats: dict[int, _RunStats] = {camera.camera_id: _RunStats() for camera in cameras}
    transitioned_spots: list[SpotState] = []
    smoother = OccupancySmoother(window=smoothing_window, on_transition=transitioned_spots.append)

    try:
        tick = 0
        while max_ticks is None or tick < max_ticks:
            ready: list[tuple[CameraRuntime, Frame, float]] = []
            for camera in cameras:
                try:
                    frame = camera.grabber.maybe_grab(now=clock())
                except Exception:
                    logger.exception(
                        "unexpected frame grab error for camera_id=%s", camera.camera_id
                    )
                    continue
                if frame is not None:
                    ready.append((camera, frame, clock()))

            if ready:
                batch = detector.predict_batch([frame for _, frame, _ in ready])
                for (camera, _frame, captured_at), detections in zip(ready, batch, strict=True):
                    tracks = camera.tracker.update(detections)
                    for raw in strategy.match(tracks, camera.spots):
                        before = len(transitioned_spots)
                        smoothed = smoother.observe(raw)
                        transitioned = len(transitioned_spots) > before
                        await persist(smoothed, transitioned)

                    camera_stats = stats[camera.camera_id]
                    camera_stats.frames_processed += 1
                    camera_stats.latencies_ms.append((clock() - captured_at) * 1000)

            tick += 1
            if max_ticks is None or tick < max_ticks:
                await sleep(tick_interval_seconds)
    finally:
        if on_run_finished is not None:
            ended_at = now_factory()
            for camera in cameras:
                camera_stats = stats[camera.camera_id]
                latencies = camera_stats.latencies_ms
                await on_run_finished(
                    PipelineRunSummary(
                        camera_id=camera.camera_id,
                        started_at=started_at,
                        ended_at=ended_at,
                        frames_processed=camera_stats.frames_processed,
                        avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
                        p95_latency_ms=_percentile(latencies, 0.95) if latencies else None,
                        model_version=model_version,
                    )
                )
