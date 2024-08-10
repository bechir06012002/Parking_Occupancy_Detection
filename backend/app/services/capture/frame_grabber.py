"""OpenCV VideoCapture adapter (Adapter pattern).

Paces itself to `sample_interval_seconds` (derived from a camera's own
`sample_fps` DB column, never a literal here) — `maybe_grab` returns None
on ticks where it isn't time to sample yet, so the batch scheduler only
collects a frame from cameras that are actually ready. Every failure mode
(bad source URI, a frame that fails to decode, an unexpected exception) is
caught here and logged with the camera_id only, never the source URI —
RTSP URLs can carry credentials and must never be logged — and turned into
a `None` return instead of propagating, so one bad camera can't
kill the batch scheduler.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import cv2

from app.services.frame import Frame

logger = logging.getLogger(__name__)


class _CaptureLike(Protocol):
    def read(self) -> tuple[bool, Frame]: ...
    def release(self) -> None: ...


class FrameGrabber:
    def __init__(
        self,
        *,
        camera_id: int,
        source_uri: str,
        sample_interval_seconds: float,
        capture_factory: Callable[[], _CaptureLike] | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._sample_interval_seconds = sample_interval_seconds
        self._capture_factory = capture_factory or (lambda: cv2.VideoCapture(source_uri))
        self._capture: _CaptureLike | None = None
        self._last_grab_at: float | None = None

    def maybe_grab(self, *, now: float | None = None) -> Frame | None:
        """Return the next frame if `sample_interval_seconds` has elapsed, else None."""
        now = time.monotonic() if now is None else now
        if (
            self._last_grab_at is not None
            and (now - self._last_grab_at) < self._sample_interval_seconds
        ):
            return None

        try:
            if self._capture is None:
                self._capture = self._capture_factory()
            ok, frame = self._capture.read()
        except Exception:
            logger.exception("frame grab failed for camera_id=%s", self._camera_id)
            return None

        if not ok:
            logger.warning("frame grab returned no frame for camera_id=%s", self._camera_id)
            return None

        self._last_grab_at = now
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
