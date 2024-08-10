"""Scale load test: many simulated cameras through one batch scheduler.

Answers one concrete question about the architecture: does the batch
scheduler in `services/pipeline/run_pipeline.py` keep *per-camera* cost
roughly flat as the camera count (and therefore the spot count) grows toward
the 200+ target? A single wide shot rarely covers 200 spots legibly, so
200+ spots means many camera feeds — and the design's bet is that folding
every camera's latest frame into one `model.predict(batch)` call per tick
amortizes the fixed inference overhead across them.

This script tests that bet with real components — the same `YoloDetector`,
`ByteTrackAdapter` (one instance per camera), `CentroidCoverageStrategy` and
`run_pipeline` Facade the worker runs, config-driven from `Settings` — but
simulates the multi-camera fleet by replaying one video file as N
independent `FrameGrabber`s, each carrying its own copy of a spot-polygon
set (ids offset per camera so they stay distinct). It sweeps a list of
camera counts and, for each, reports the mean per-camera avg/p95 latency
`run_pipeline` recorded plus the wall-clock time per scheduler tick.

Persistence here is an in-memory no-op: the DB-write leg of the latency
budget is already measured against real Postgres by
`scripts/benchmark_latency.py`; this script isolates scheduler +
batched-inference cost as camera count scales. Pass `--with-db` to
additionally exercise real `record_spot_observation` writes (needs a
running Postgres per `backend/.env`).

This is a standalone script, not a pytest test — it needs a real video and
real model weights.

Usage:
    uv run python scripts/load_test_scale.py \
        --video ../data/parking_lot.mov \
        --spots-json scripts/ground_truth/parking_lot.json \
        --camera-counts 1,4,8,12 \
        --ticks-per-count 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.domain.occupancy import Spot, SpotState
from app.domain.pipeline import PipelineRunSummary
from app.services.capture.frame_grabber import FrameGrabber
from app.services.detection.yolo import YoloDetector
from app.services.occupancy.matching import CentroidCoverageStrategy
from app.services.pipeline.run_pipeline import CameraRuntime, run_pipeline
from app.services.tracking.byte_track import ByteTrackAdapter


def _load_polygons(path: Path) -> list[tuple[tuple[float, float], ...]]:
    data = json.loads(path.read_text())
    return [tuple((float(p[0]), float(p[1])) for p in entry["polygon"]) for entry in data["spots"]]


def _spots_for_camera(
    polygons: list[tuple[tuple[float, float], ...]], camera_index: int
) -> list[Spot]:
    # Offset spot ids per camera so every simulated spot is globally distinct,
    # exactly as real per-camera rows from `parking_spots` would be.
    base = camera_index * 1000
    return [Spot(spot_id=base + i, polygon=poly) for i, poly in enumerate(polygons)]


async def _run_one_sweep(
    *,
    camera_count: int,
    polygons: list[tuple[tuple[float, float], ...]],
    video: Path,
    detector: YoloDetector,
    ticks: int,
    persist: object,
) -> dict[str, float]:
    settings = get_settings()
    strategy = CentroidCoverageStrategy(coverage_threshold=settings.occupancy_coverage_threshold)

    cameras = [
        CameraRuntime(
            camera_id=camera_index,
            spots=_spots_for_camera(polygons, camera_index),
            grabber=FrameGrabber(
                camera_id=camera_index,
                source_uri=str(video),
                sample_interval_seconds=0.0,
            ),
            tracker=ByteTrackAdapter(),
        )
        for camera_index in range(camera_count)
    ]

    summaries: list[PipelineRunSummary] = []

    async def on_run_finished(summary: PipelineRunSummary) -> None:
        summaries.append(summary)

    wall_start = time.monotonic()
    await run_pipeline(
        cameras=cameras,
        detector=detector,
        strategy=strategy,
        smoothing_window=settings.smoothing_window,
        persist=persist,  # type: ignore[arg-type]
        model_version=settings.yolo_weights_path,
        tick_interval_seconds=0.0,
        on_run_finished=on_run_finished,
        max_ticks=ticks,
    )
    wall_elapsed = time.monotonic() - wall_start

    for camera in cameras:
        camera.grabber.release()

    avg_latencies = [s.avg_latency_ms for s in summaries if s.avg_latency_ms is not None]
    p95_latencies = [s.p95_latency_ms for s in summaries if s.p95_latency_ms is not None]
    frames = sum(s.frames_processed for s in summaries)

    return {
        "camera_count": camera_count,
        "total_spots": camera_count * len(polygons),
        "mean_avg_latency_ms": sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0.0,
        "mean_p95_latency_ms": sum(p95_latencies) / len(p95_latencies) if p95_latencies else 0.0,
        "wall_ms_per_tick": (wall_elapsed / ticks) * 1000 if ticks else 0.0,
        "frames_processed": float(frames),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--spots-json", required=True, type=Path)
    parser.add_argument("--camera-counts", default="1,4,8,12")
    parser.add_argument("--ticks-per-count", type=int, default=20)
    parser.add_argument(
        "--with-db",
        action="store_true",
        help="Also run real record_spot_observation writes (needs a running Postgres)",
    )
    args = parser.parse_args()

    settings = get_settings()
    polygons = _load_polygons(args.spots_json)
    counts = [int(c) for c in args.camera_counts.split(",") if c.strip()]

    detector = YoloDetector(
        weights_path=settings.yolo_weights_path,
        confidence_threshold=settings.yolo_conf_threshold,
        imgsz=settings.yolo_imgsz,
    )

    # Warm up the model once so first-tick CUDA-graph / lazy-init cost
    # (docs/evaluation.md notes ~1.3s) doesn't land inside a measured sweep.
    import cv2

    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {args.video}")
    detector.predict_batch([frame])

    if args.with_db:
        from app.db.session import async_session_factory
        from app.db.writer import record_spot_observation

        session_cm = async_session_factory()
        session = await session_cm.__aenter__()

        async def persist(state: SpotState, transitioned: bool) -> None:
            await record_spot_observation(
                session, state, transitioned=transitioned, now=datetime.now(UTC)
            )
            await session.commit()
    else:
        session_cm = None

        async def persist(state: SpotState, transitioned: bool) -> None:
            return None

    results = []
    try:
        for count in counts:
            result = await _run_one_sweep(
                camera_count=count,
                polygons=polygons,
                video=args.video,
                detector=detector,
                ticks=args.ticks_per_count,
                persist=persist,
            )
            results.append(result)
            print(
                f"cameras={result['camera_count']:>3}  "
                f"spots={result['total_spots']:>4}  "
                f"per-camera avg={result['mean_avg_latency_ms']:>7.1f}ms  "
                f"per-camera p95={result['mean_p95_latency_ms']:>7.1f}ms  "
                f"wall/tick={result['wall_ms_per_tick']:>7.1f}ms"
            )
    finally:
        if session_cm is not None:
            await session_cm.__aexit__(None, None, None)

    print("\n--- summary ---")
    print("budget: per-camera end-to-end <= 2000ms")
    if len(results) >= 2:
        first, last = results[0], results[-1]
        spot_growth = last["total_spots"] / max(1.0, first["total_spots"])
        lat_growth = last["mean_avg_latency_ms"] / max(1e-6, first["mean_avg_latency_ms"])
        print(
            f"spot count grew {spot_growth:.1f}x "
            f"({first['total_spots']:.0f} -> {last['total_spots']:.0f}); "
            f"per-camera avg latency grew {lat_growth:.2f}x "
            f"({first['mean_avg_latency_ms']:.0f} -> {last['mean_avg_latency_ms']:.0f}ms)"
        )
    max_p95 = max((r["mean_p95_latency_ms"] for r in results), default=0.0)
    print(
        f"worst per-camera p95 across all sweeps: {max_p95:.0f}ms "
        f"({'within' if max_p95 <= 2000 else 'OVER'} budget)"
    )


if __name__ == "__main__":
    asyncio.run(main())
