"""Worker entrypoint — runs services/pipeline continuously.

A standalone process, never FastAPI BackgroundTasks: reads active cameras
(and their spots, via the same camera_id-scoped db/scoped.py helper the API
uses) from Postgres at startup, loads one YOLO model and builds one
FrameGrabber + one ByteTrackAdapter per camera, then hands everything to
the pipeline Facade. `--max-ticks` is only for end-to-end verification
runs — real deployment omits it and runs until killed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Camera
from app.db.scoped import spots_for_camera
from app.db.session import async_session_factory
from app.db.writer import record_pipeline_run, record_spot_observation
from app.domain.occupancy import Spot, SpotState
from app.domain.pipeline import PipelineRunSummary
from app.services.capture.frame_grabber import FrameGrabber
from app.services.detection.yolo import YoloDetector
from app.services.occupancy.matching import CentroidCoverageStrategy
from app.services.pipeline.run_pipeline import CameraRuntime, run_pipeline
from app.services.tracking.byte_track import ByteTrackAdapter

logger = logging.getLogger(__name__)


async def _load_camera_runtimes(session: AsyncSession) -> list[CameraRuntime]:
    cameras = (await session.execute(select(Camera).where(Camera.is_active.is_(True)))).scalars()

    runtimes: list[CameraRuntime] = []
    for camera in cameras:
        spot_rows = (await session.execute(spots_for_camera(camera.id))).scalars()
        spots = [
            Spot(spot_id=row.id, polygon=tuple((p[0], p[1]) for p in row.polygon))
            for row in spot_rows
        ]
        runtimes.append(
            CameraRuntime(
                camera_id=camera.id,
                spots=spots,
                grabber=FrameGrabber(
                    camera_id=camera.id,
                    source_uri=camera.source_uri,
                    sample_interval_seconds=1.0 / camera.sample_fps,
                ),
                tracker=ByteTrackAdapter(),
            )
        )
    return runtimes


async def run(*, max_ticks: int | None) -> None:
    settings = get_settings()

    async with async_session_factory() as session:
        cameras = await _load_camera_runtimes(session)
        if not cameras:
            logger.warning("no active cameras found — nothing to do")
            return

        detector = YoloDetector(
            weights_path=settings.yolo_weights_path,
            confidence_threshold=settings.yolo_conf_threshold,
            imgsz=settings.yolo_imgsz,
        )
        strategy = CentroidCoverageStrategy(
            coverage_threshold=settings.occupancy_coverage_threshold
        )

        async def persist(state: SpotState, transitioned: bool) -> None:
            await record_spot_observation(
                session, state, transitioned=transitioned, now=datetime.now(UTC)
            )
            await session.commit()

        async def on_run_finished(summary: PipelineRunSummary) -> None:
            await record_pipeline_run(session, summary)
            await session.commit()

        await run_pipeline(
            cameras=cameras,
            detector=detector,
            strategy=strategy,
            smoothing_window=settings.smoothing_window,
            persist=persist,
            model_version=settings.yolo_weights_path,
            tick_interval_seconds=1.0 / settings.sample_fps,
            on_run_finished=on_run_finished,
            max_ticks=max_ticks,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Run this many scheduler ticks then stop (omit to run continuously)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(max_ticks=args.max_ticks))
    except KeyboardInterrupt:
        logger.info("worker stopped")


if __name__ == "__main__":
    main()
