from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    """One vehicle detection with a stable identity across frames.

    track_id is only meaningful within a single camera's frame sequence —
    one ByteTrack instance is created per camera, never shared.
    """

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
