"""Latency benchmark: end-to-end frame-capture -> DB-write timing.

Seeds a dedicated eval camera (idempotent) from the same ground-truth spot
polygons evaluate_accuracy.py uses, then runs real detect -> track -> match
-> persist against a real video for `--max-ticks` ticks — the same adapters
run_pipeline.py uses, config-driven from Settings, against a real Postgres
connection — and reports p50/p95 latency against the <=2s budget.

Every observation is persisted via the full upsert + occupancy_events path
(transitioned=True), not the lighter last-seen-at-only touch a real worker
uses once a spot's state has settled — that's a deliberate conservative
(upper-bound) choice for a latency number, not an oversight.

This is a standalone script, not a pytest test — it needs a real video,
real model weights, and a real Postgres connection.

Usage:
    uv run python scripts/benchmark_latency.py \
        --video ../data/parking_lot.mov \
        --ground-truth scripts/ground_truth/parking_lot.json \
        --camera-name eval-lot-1 \
        --max-ticks 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from annotate_spots import SpotAnnotation, get_or_create_camera, save_spots
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import ParkingSpot
from app.db.session import async_session_factory
from app.db.writer import record_pipeline_run, record_spot_observation
from app.domain.occupancy import Spot
from app.domain.pipeline import PipelineRunSummary
from app.services.capture.frame_grabber import FrameGrabber
from app.services.detection.yolo import YoloDetector
from app.services.occupancy.matching import CentroidCoverageStrategy
from app.services.tracking.byte_track import ByteTrackAdapter


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--camera-name", default="eval-lot-1")
    parser.add_argument("--max-ticks", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    gt_data = json.loads(args.ground_truth.read_text())

    async with async_session_factory() as session:
        camera = await get_or_create_camera(
            session,
            name=args.camera_name,
            source_uri=str(args.video),
            resolution="1920x1080",
            sample_fps=1.0,
        )
        existing = (
            await session.execute(
                select(ParkingSpot.id).where(ParkingSpot.camera_id == camera.id).limit(1)
            )
        ).scalar_one_or_none()
        if existing is None:
            spot_annotations = [
                SpotAnnotation(label=e["label"], polygon=e["polygon"]) for e in gt_data["spots"]
            ]
            await save_spots(session, camera_id=camera.id, spots=spot_annotations)

        spot_rows = (
            (await session.execute(select(ParkingSpot).where(ParkingSpot.camera_id == camera.id)))
            .scalars()
            .all()
        )
        spots = [
            Spot(spot_id=row.id, polygon=tuple((p[0], p[1]) for p in row.polygon))
            for row in spot_rows
        ]

        detector = YoloDetector(
            weights_path=settings.yolo_weights_path,
            confidence_threshold=settings.yolo_conf_threshold,
            imgsz=settings.yolo_imgsz,
        )
        tracker = ByteTrackAdapter()
        strategy = CentroidCoverageStrategy(
            coverage_threshold=settings.occupancy_coverage_threshold
        )
        grabber = FrameGrabber(
            camera_id=camera.id, source_uri=str(args.video), sample_interval_seconds=0.0
        )

        latencies_ms: list[float] = []
        started_at = datetime.now(UTC)
        frames_processed = 0

        for _ in range(args.max_ticks):
            captured_at = time.monotonic()
            frame = grabber.maybe_grab(now=captured_at)
            if frame is None:
                continue
            [detections] = detector.predict_batch([frame])
            tracks = tracker.update(detections)
            for raw in strategy.match(tracks, spots):
                await record_spot_observation(
                    session, raw, transitioned=True, now=datetime.now(UTC)
                )
            await session.commit()
            frames_processed += 1
            latencies_ms.append((time.monotonic() - captured_at) * 1000)

        ended_at = datetime.now(UTC)
        avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else None
        p95_ms = _percentile(latencies_ms, 0.95) if latencies_ms else None
        await record_pipeline_run(
            session,
            PipelineRunSummary(
                camera_id=camera.id,
                started_at=started_at,
                ended_at=ended_at,
                frames_processed=frames_processed,
                avg_latency_ms=avg_ms,
                p95_latency_ms=p95_ms,
                model_version=settings.yolo_weights_path,
            ),
        )
        await session.commit()

    p50_ms = _percentile(latencies_ms, 0.5) if latencies_ms else None
    print(f"video: {args.video}")
    print(f"frames processed: {frames_processed}")
    print(f"per-frame latencies (ms): {[round(x) for x in latencies_ms]}")
    print(f"p50 latency: {p50_ms:.0f}ms" if p50_ms is not None else "p50 latency: n/a")
    print(f"p95 latency: {p95_ms:.0f}ms" if p95_ms is not None else "p95 latency: n/a")
    print("budget: <=2000ms")
    if p95_ms is not None:
        print(f"p95 within budget: {p95_ms <= 2000}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
