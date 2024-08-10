import pytest

from app.domain.occupancy import SpotState
from app.services.occupancy.state import OccupancySmoother


def _raw(
    spot_id: int, is_occupied: bool, *, confidence: float = 0.9, track_id: int | None = 5
) -> SpotState:
    return SpotState(
        spot_id=spot_id,
        is_occupied=is_occupied,
        confidence=confidence,
        track_id=track_id if is_occupied else None,
    )


def test_window_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        OccupancySmoother(window=0)


def test_clean_occupied_settles_occupied() -> None:
    smoother = OccupancySmoother(window=5)

    results = [smoother.observe(_raw(1, True)) for _ in range(5)]

    assert all(r.is_occupied for r in results)


def test_clean_free_settles_free() -> None:
    smoother = OccupancySmoother(window=5)

    results = [smoother.observe(_raw(1, False)) for _ in range(5)]

    assert all(not r.is_occupied for r in results)


def test_single_frame_flicker_does_not_flip_state() -> None:
    transitions: list[SpotState] = []
    smoother = OccupancySmoother(window=5, on_transition=transitions.append)

    for _ in range(5):
        smoother.observe(_raw(1, False))
    assert len(transitions) == 1  # initial establishment as free

    result = smoother.observe(_raw(1, True))  # one-frame blip
    smoother.observe(_raw(1, False))
    smoother.observe(_raw(1, False))

    assert result.is_occupied is False
    assert len(transitions) == 1  # no additional transition fired


def test_sustained_change_flips_after_enough_samples() -> None:
    transitions: list[SpotState] = []
    smoother = OccupancySmoother(window=5, on_transition=transitions.append)

    for _ in range(5):
        smoother.observe(_raw(1, False))
    assert len(transitions) == 1

    results = [smoother.observe(_raw(1, True)) for _ in range(5)]

    assert results[-1].is_occupied is True
    assert any(r.is_occupied for r in results)  # flips at some point within the window
    assert len(transitions) == 2  # exactly one more transition: free -> occupied


def test_on_transition_fires_once_per_flip_not_per_observation() -> None:
    transitions: list[SpotState] = []
    smoother = OccupancySmoother(window=3, on_transition=transitions.append)

    for _ in range(10):
        smoother.observe(_raw(1, True))

    assert len(transitions) == 1


def test_spots_are_tracked_independently() -> None:
    smoother = OccupancySmoother(window=3)

    for _ in range(3):
        smoother.observe(_raw(1, True))
    result_2 = smoother.observe(_raw(2, False))

    assert result_2.is_occupied is False


def test_observe_returns_current_confidence_and_track_id() -> None:
    smoother = OccupancySmoother(window=3)

    result = smoother.observe(_raw(1, True, confidence=0.77, track_id=99))

    assert result.confidence == 0.77
    assert result.track_id == 99
