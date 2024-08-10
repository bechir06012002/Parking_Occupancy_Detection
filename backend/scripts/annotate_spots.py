"""Interactive (or scripted) parking-spot polygon annotation tool.

Draws polygons over a camera's first frame and writes them into
`parking_spots`, creating the `cameras` row if it doesn't exist yet. Writes
straight to the DB rather than going through the API.

Interactive usage (opens a window; requires a display):
    uv run python scripts/annotate_spots.py --video data/lot1.mp4 \
        --camera-name lot1

Controls: left-click to add a polygon point, 'n' to finish the current
polygon and label it, 'u' to undo the last point, 's' to save and exit,
'q'/Esc to quit without saving.

Scripted usage (no display needed — e.g. seeding, see seed_demo.py):
    uv run python scripts/annotate_spots.py --video data/lot1.mp4 \
        --camera-name lot1 --spots-json spots.json

`spots.json` is a list of {"label": str, "polygon": [[x, y], ...]}.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, ParkingSpot
from app.db.session import async_session_factory

Frame = MatLike


@dataclass(frozen=True)
class SpotAnnotation:
    label: str
    polygon: list[list[float]]


def load_first_frame(video_path: Path) -> tuple[Frame, str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame from: {video_path}")
        return frame, f"{width}x{height}"
    finally:
        capture.release()


class _InteractiveAnnotator:
    """Owns the OpenCV window/mouse-callback state for one annotation session."""

    def __init__(self, frame: Frame) -> None:
        self._base_frame = frame
        self._finished: list[SpotAnnotation] = []
        self._current_points: list[list[float]] = []
        self._window_name = "annotate_spots — n: finish spot, u: undo, s: save, q: quit"

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_points.append([float(x), float(y)])

    def _render(self) -> Frame:
        canvas = self._base_frame.copy()
        for spot in self._finished:
            points = np.array(spot.polygon, dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=True, color=(0, 200, 0), thickness=2)
            cv2.putText(
                canvas,
                spot.label,
                (int(points[0][0]), int(points[0][1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                2,
            )
        for point in self._current_points:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
        if len(self._current_points) > 1:
            points = np.array(self._current_points, dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=False, color=(0, 0, 255), thickness=1)
        return canvas

    def run(self) -> list[SpotAnnotation]:
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._on_mouse)
        try:
            while True:
                cv2.imshow(self._window_name, self._render())
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):  # Esc
                    return []
                if key == ord("u") and self._current_points:
                    self._current_points.pop()
                elif key == ord("n") and len(self._current_points) >= 3:
                    label = input(f"Label for spot #{len(self._finished) + 1}: ").strip()
                    self._finished.append(
                        SpotAnnotation(
                            label=label or f"spot-{len(self._finished) + 1}",
                            polygon=self._current_points,
                        )
                    )
                    self._current_points = []
                elif key == ord("s"):
                    return self._finished
        finally:
            cv2.destroyWindow(self._window_name)


def load_spots_from_json(path: Path) -> list[SpotAnnotation]:
    raw = json.loads(path.read_text())
    return [SpotAnnotation(label=item["label"], polygon=item["polygon"]) for item in raw]


async def get_or_create_camera(
    session: AsyncSession, *, name: str, source_uri: str, resolution: str, sample_fps: float
) -> Camera:
    existing = await session.execute(select(Camera).where(Camera.name == name))
    camera = existing.scalar_one_or_none()
    if camera is not None:
        return camera
    camera = Camera(
        name=name,
        source_uri=source_uri,
        resolution=resolution,
        sample_fps=sample_fps,
        is_active=True,
    )
    session.add(camera)
    await session.flush()
    return camera


async def save_spots(session: AsyncSession, *, camera_id: int, spots: list[SpotAnnotation]) -> None:
    for index, spot in enumerate(spots):
        session.add(
            ParkingSpot(
                camera_id=camera_id,
                label=spot.label,
                polygon=spot.polygon,
                spot_index=index,
            )
        )
    await session.commit()


async def annotate(
    *,
    video_path: Path,
    camera_name: str,
    sample_fps: float,
    source_uri: str | None,
    spots: list[SpotAnnotation],
    resolution: str,
) -> None:
    async with async_session_factory() as session:
        camera = await get_or_create_camera(
            session,
            name=camera_name,
            source_uri=source_uri or str(video_path),
            resolution=resolution,
            sample_fps=sample_fps,
        )
        await save_spots(session, camera_id=camera.id, spots=spots)
    print(f"Saved {len(spots)} spot(s) for camera '{camera_name}' (id={camera.id}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--source-uri", default=None, help="Defaults to --video path")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--spots-json",
        type=Path,
        default=None,
        help="Skip the interactive window and load polygons from this JSON file instead",
    )
    args = parser.parse_args()

    frame, resolution = load_first_frame(args.video)

    if args.spots_json is not None:
        spots = load_spots_from_json(args.spots_json)
    else:
        spots = _InteractiveAnnotator(frame).run()

    if not spots:
        print("No spots to save — exiting without writing to the DB.")
        sys.exit(1)

    asyncio.run(
        annotate(
            video_path=args.video,
            camera_name=args.camera_name,
            sample_fps=args.sample_fps,
            source_uri=args.source_uri,
            spots=spots,
            resolution=resolution,
        )
    )


if __name__ == "__main__":
    main()
