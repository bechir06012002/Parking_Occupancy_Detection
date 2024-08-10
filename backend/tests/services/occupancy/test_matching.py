from app.domain.occupancy import Spot
from app.domain.tracking import Track
from app.services.occupancy.matching import CentroidCoverageStrategy, CentroidIoUStrategy

SPOT = Spot(spot_id=1, polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))


def _strategy(iou_threshold: float = 0.3) -> CentroidIoUStrategy:
    return CentroidIoUStrategy(iou_threshold=iou_threshold)


def test_clean_occupied_track_matching_spot_bbox() -> None:
    track = Track(track_id=42, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, class_name="car")

    [result] = _strategy().match([track], [SPOT])

    assert result.spot_id == 1
    assert result.is_occupied is True
    assert result.track_id == 42
    assert result.confidence == 0.9


def test_clean_free_with_no_tracks() -> None:
    [result] = _strategy().match([], [SPOT])

    assert result.is_occupied is False
    assert result.track_id is None
    assert result.confidence == 0.0


def test_free_when_track_is_far_from_spot() -> None:
    track = Track(track_id=1, bbox=(500.0, 500.0, 510.0, 510.0), confidence=0.9, class_name="car")

    [result] = _strategy().match([track], [SPOT])

    assert result.is_occupied is False


def test_free_when_centroid_inside_but_iou_below_threshold() -> None:
    # Centroid (5, 5) sits inside SPOT's polygon, but the track's bbox is
    # huge relative to the spot, so IoU is tiny — the AND rule must reject it.
    track = Track(track_id=7, bbox=(-95.0, -95.0, 105.0, 105.0), confidence=0.9, class_name="car")

    [result] = _strategy(iou_threshold=0.3).match([track], [SPOT])

    assert result.is_occupied is False


def test_free_when_iou_high_but_centroid_outside_polygon() -> None:
    # bbox overlaps SPOT substantially but its centroid falls outside the
    # polygon — centroid check must reject it even with strong IoU.
    track = Track(track_id=3, bbox=(9.0, -100.0, 11.0, 9.0), confidence=0.9, class_name="car")

    [result] = _strategy(iou_threshold=0.01).match([track], [SPOT])

    assert result.is_occupied is False


def test_iou_exactly_at_threshold_counts() -> None:
    track = Track(track_id=9, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.5, class_name="car")
    strategy = CentroidIoUStrategy(iou_threshold=1.0)  # perfect overlap == 1.0

    [result] = strategy.match([track], [SPOT])

    assert result.is_occupied is True


def test_best_of_multiple_qualifying_tracks_is_selected() -> None:
    worse = Track(track_id=1, bbox=(0.0, 0.0, 6.0, 6.0), confidence=0.4, class_name="car")
    better = Track(track_id=2, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, class_name="car")

    [result] = _strategy().match([worse, better], [SPOT])

    assert result.track_id == 2


def test_multiple_spots_matched_independently() -> None:
    other_spot = Spot(spot_id=2, polygon=((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)))
    track_in_spot_one = Track(
        track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, class_name="car"
    )

    results = _strategy().match([track_in_spot_one], [SPOT, other_spot])

    by_spot = {r.spot_id: r for r in results}
    assert by_spot[1].is_occupied is True
    assert by_spot[2].is_occupied is False


# --- CentroidCoverageStrategy (the default) ---

# A stall drawn much longer than the car it holds: the polygon is 10 wide x
# 40 tall, a well-parked car fills only a middle 10x16. IoU(car, stall) is
# 0.4 — below a 0.5 gate — but 100% of the car is inside the stall, so
# coverage sees it as occupied. This is the case the default strategy exists
# for.
TALL_SPOT = Spot(spot_id=5, polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 40.0), (0.0, 40.0)))


def test_coverage_occupied_when_whole_vehicle_sits_in_a_longer_stall() -> None:
    car = Track(track_id=1, bbox=(0.0, 12.0, 10.0, 28.0), confidence=0.8, class_name="car")

    [result] = CentroidCoverageStrategy(coverage_threshold=0.5).match([car], [TALL_SPOT])

    assert result.is_occupied is True
    assert result.track_id == 1
    assert CentroidIoUStrategy(iou_threshold=0.5).match([car], [TALL_SPOT])[0].is_occupied is False


def test_coverage_free_when_vehicle_hangs_well_outside_the_stall() -> None:
    # Centroid (2, 14) is inside, but the car is 16 wide and only 10 of that
    # overlaps the stall in x: coverage is 0.625, below a 0.7 gate.
    car = Track(track_id=2, bbox=(-6.0, 8.0, 10.0, 20.0), confidence=0.9, class_name="car")

    [result] = CentroidCoverageStrategy(coverage_threshold=0.7).match([car], [TALL_SPOT])

    assert result.is_occupied is False


def test_coverage_free_with_no_tracks() -> None:
    [result] = CentroidCoverageStrategy(coverage_threshold=0.5).match([], [TALL_SPOT])

    assert result.is_occupied is False
    assert result.confidence == 0.0
