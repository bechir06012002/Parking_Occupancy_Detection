"""Temporal smoothing / majority-vote hysteresis (Observer pattern).

A spot's persisted state only flips once the new value holds a majority of
the last `window` raw observations — this absorbs single-frame
missed detections/track-ID switches without hiding a real, sustained
change. `window` is a required constructor parameter, never a literal here.

`on_transition` is the Observer hook: fired whenever a spot's state is
first established or actually flips, so the caller (the worker's async DB
writer, in P4) can insert an occupancy_events row / update the in-memory
cache without this module knowing anything about persistence.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from app.domain.occupancy import SpotState


class OccupancySmoother:
    def __init__(
        self, *, window: int, on_transition: Callable[[SpotState], None] | None = None
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._on_transition = on_transition
        self._history: dict[int, deque[bool]] = {}
        self._state: dict[int, SpotState] = {}

    def observe(self, raw: SpotState) -> SpotState:
        """Fold one more raw observation into the spot's history; return its smoothed state."""
        history = self._history.setdefault(raw.spot_id, deque(maxlen=self._window))
        history.append(raw.is_occupied)
        majority_occupied = sum(history) * 2 > len(history)

        smoothed = SpotState(
            spot_id=raw.spot_id,
            is_occupied=majority_occupied,
            confidence=raw.confidence,
            track_id=raw.track_id,
        )
        previous = self._state.get(raw.spot_id)
        self._state[raw.spot_id] = smoothed
        if (previous is None or previous.is_occupied != majority_occupied) and (
            self._on_transition is not None
        ):
            self._on_transition(smoothed)
        return smoothed
