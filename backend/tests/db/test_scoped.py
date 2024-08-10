from app.db import scoped


def test_spots_for_camera_filters_on_camera_id() -> None:
    compiled = str(scoped.spots_for_camera(42))

    assert "parking_spots.camera_id = " in compiled


def test_pipeline_runs_for_camera_filters_on_camera_id() -> None:
    compiled = str(scoped.pipeline_runs_for_camera(42))

    assert "pipeline_runs.camera_id = " in compiled


def test_occupancy_state_for_camera_joins_through_parking_spot() -> None:
    compiled = str(scoped.occupancy_state_for_camera(42))

    assert "JOIN parking_spots" in compiled
    assert "parking_spots.camera_id = " in compiled


def test_occupancy_events_for_camera_joins_through_parking_spot() -> None:
    compiled = str(scoped.occupancy_events_for_camera(42))

    assert "JOIN parking_spots" in compiled
    assert "parking_spots.camera_id = " in compiled


def test_camera_by_id_filters_on_id() -> None:
    compiled = str(scoped.camera_by_id(42))

    assert "cameras.id = " in compiled
