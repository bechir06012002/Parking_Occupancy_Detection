"""ByteTrack tracking adapter (Adapter pattern).

Isolates the supervision.ByteTrack call boundary. One `ByteTrackAdapter`
instance must be created per camera and kept alive across ticks — track IDs
are only meaningful within a single camera's frame sequence, so one
instance must never be shared across cameras. That lifecycle is the caller's
(the worker's, in P4) responsibility; this class does not know about
cameras at all.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import supervision as sv

from app.domain.detection import Detection
from app.domain.tracking import Track


class _TrackerLike(Protocol):
    def update_with_detections(self, detections: sv.Detections) -> sv.Detections: ...


def _to_sv_detections(detections: list[Detection]) -> sv.Detections:
    # Always shape xyxy as (n, 4) and always populate the "class_name" data
    # key, even for n=0 — a bare sv.Detections.empty() has no "class_name"
    # key at all, and ByteTrack can hand back a zero-length (but non-None)
    # tracker_id here, so _to_tracks must not assume the key exists either.
    return sv.Detections(
        xyxy=np.array([d.bbox for d in detections], dtype=np.float32).reshape(-1, 4),
        confidence=np.array([d.confidence for d in detections], dtype=np.float32),
        class_id=np.zeros(len(detections), dtype=int),
        data={"class_name": np.array([d.class_name for d in detections], dtype=object)},
    )


def _to_tracks(tracked: sv.Detections) -> list[Track]:
    if len(tracked) == 0 or tracked.tracker_id is None:
        return []
    class_names = tracked.data.get("class_name")
    tracks: list[Track] = []
    for i in range(len(tracked)):
        x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
        confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
        class_name = str(class_names[i]) if class_names is not None else ""
        tracks.append(
            Track(
                track_id=int(tracked.tracker_id[i]),
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                class_name=class_name,
            )
        )
    return tracks


class ByteTrackAdapter:
    """Wraps one supervision.ByteTrack instance for a single camera's stream."""

    def __init__(self, *, tracker: _TrackerLike | None = None) -> None:
        self._tracker: _TrackerLike = tracker if tracker is not None else sv.ByteTrack()

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker by one frame's detections and return current tracks."""
        tracked = self._tracker.update_with_detections(_to_sv_detections(detections))
        return _to_tracks(tracked)
