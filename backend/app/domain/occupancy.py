from dataclasses import dataclass


@dataclass(frozen=True)
class Spot:
    """A parking spot's polygon geometry, decoupled from the DB row."""

    spot_id: int
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SpotState:
    """One spot's occupancy at a point in time.

    Shape is shared by both matching.py's raw per-tick reading and
    state.py's temporally-smoothed, persisted state — same fields, either
    a single observation or the majority-vote result of many.
    """

    spot_id: int
    is_occupied: bool
    confidence: float
    track_id: int | None
