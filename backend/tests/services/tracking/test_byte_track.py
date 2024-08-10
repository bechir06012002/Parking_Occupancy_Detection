import numpy as np
import pytest
import supervision as sv

from app.domain.detection import Detection
from app.services.tracking.byte_track import ByteTrackAdapter


class FakeTracker:
    """Assigns track_ids from a fixed, pre-scripted sequence per call."""

    def __init__(self, track_ids_by_call: list[list[int]]) -> None:
        self._track_ids_by_call = track_ids_by_call
        self.calls: list[sv.Detections] = []

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        self.calls.append(detections)
        if len(detections) == 0:
            return detections
        track_ids = self._track_ids_by_call[len(self.calls) - 1]
        detections.tracker_id = np.array(track_ids, dtype=int)
        return detections


def test_update_returns_tracks_with_assigned_ids() -> None:
    tracker = FakeTracker(track_ids_by_call=[[7, 8]])
    adapter = ByteTrackAdapter(tracker=tracker)
    detections = [
        Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_name="car"),
        Detection(bbox=(20, 20, 30, 30), confidence=0.8, class_name="truck"),
    ]

    tracks = adapter.update(detections)

    assert [t.track_id for t in tracks] == [7, 8]
    assert [t.class_name for t in tracks] == ["car", "truck"]
    assert tracks[0].bbox == (0.0, 0.0, 10.0, 10.0)
    assert tracks[1].confidence == pytest.approx(0.8)


def test_update_with_no_detections_returns_no_tracks() -> None:
    tracker = FakeTracker(track_ids_by_call=[])
    adapter = ByteTrackAdapter(tracker=tracker)

    tracks = adapter.update([])

    assert tracks == []


class EmptyDataTracker:
    """Reproduces a real supervision.ByteTrack quirk: on a zero-detection
    call it returns a non-None-but-empty tracker_id array and a `data` dict
    with no "class_name" key at all (unlike a fresh Detections.empty(),
    which has tracker_id=None) — this crashed _to_tracks with a bare
    `tracked.data["class_name"]` lookup until it switched to `.get()`.
    """

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=int),
            tracker_id=np.zeros((0,), dtype=int),
            data={},
        )


def test_update_with_no_detections_and_empty_non_none_tracker_id_does_not_crash() -> None:
    adapter = ByteTrackAdapter(tracker=EmptyDataTracker())

    tracks = adapter.update([])

    assert tracks == []


def test_same_tracker_instance_is_reused_across_calls() -> None:
    tracker = FakeTracker(track_ids_by_call=[[1], [1]])
    adapter = ByteTrackAdapter(tracker=tracker)
    detection = Detection(bbox=(0, 0, 1, 1), confidence=0.5, class_name="car")

    adapter.update([detection])
    adapter.update([detection])

    assert len(tracker.calls) == 2
