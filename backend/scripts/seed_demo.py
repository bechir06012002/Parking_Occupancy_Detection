"""Seed one demo camera and its annotated spots.

Generates a small synthetic placeholder video under data/ (real footage is
only needed once the pipeline and evaluation scripts run against it) and
writes a fixed demo camera + 4 spot polygons via the same persistence
functions annotate_spots.py uses.

Usage: uv run python scripts/seed_demo.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
from annotate_spots import SpotAnnotation, annotate
from sqlalchemy import select

from app.db.models import Camera, ParkingSpot
from app.db.session import async_session_factory

DEMO_VIDEO_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "demo_camera.mp4"
FRAME_SIZE = (640, 480)  # width, height

DEMO_SPOTS = [
    SpotAnnotation(
        label="A1", polygon=[[40.0, 40.0], [180.0, 40.0], [180.0, 200.0], [40.0, 200.0]]
    ),
    SpotAnnotation(
        label="A2", polygon=[[200.0, 40.0], [340.0, 40.0], [340.0, 200.0], [200.0, 200.0]]
    ),
    SpotAnnotation(
        label="A3", polygon=[[40.0, 240.0], [180.0, 240.0], [180.0, 400.0], [40.0, 400.0]]
    ),
    SpotAnnotation(
        label="A4", polygon=[[200.0, 240.0], [340.0, 240.0], [340.0, 400.0], [200.0, 400.0]]
    ),
]


def make_demo_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, FRAME_SIZE)
    try:
        for i in range(30):
            frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 200, dtype=np.uint8)
            cv2.putText(
                frame,
                f"SYNTHETIC PLACEHOLDER frame {i}",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()


async def demo_camera_already_seeded() -> bool:
    async with async_session_factory() as session:
        camera = (
            await session.execute(select(Camera).where(Camera.name == "demo-lot-1"))
        ).scalar_one_or_none()
        if camera is None:
            return False
        existing_spot = (
            await session.execute(
                select(ParkingSpot.id).where(ParkingSpot.camera_id == camera.id).limit(1)
            )
        ).scalar_one_or_none()
        return existing_spot is not None


async def main() -> None:
    if await demo_camera_already_seeded():
        print("demo-lot-1 already has spots seeded — nothing to do.")
        return

    if not DEMO_VIDEO_PATH.exists():
        make_demo_video(DEMO_VIDEO_PATH)
        print(f"Generated synthetic placeholder video: {DEMO_VIDEO_PATH}")

    await annotate(
        video_path=DEMO_VIDEO_PATH,
        camera_name="demo-lot-1",
        sample_fps=1.0,
        source_uri=str(DEMO_VIDEO_PATH),
        spots=DEMO_SPOTS,
        resolution=f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
    )


if __name__ == "__main__":
    asyncio.run(main())
