from sqlalchemy import Select

from app.db.models import Camera, OccupancyEvent, OccupancyState, ParkingSpot, PipelineRun


def spots_for_camera(camera_id: int) -> Select[tuple[ParkingSpot]]:
    """Select statement for a single camera's own spots."""
    return Select(ParkingSpot).where(ParkingSpot.camera_id == camera_id)


def pipeline_runs_for_camera(camera_id: int) -> Select[tuple[PipelineRun]]:
    """Select statement for a single camera's own pipeline runs."""
    return Select(PipelineRun).where(PipelineRun.camera_id == camera_id)


def occupancy_state_for_camera(camera_id: int) -> Select[tuple[OccupancyState]]:
    """Select statement for the current occupancy state of one camera's spots.

    OccupancyState has no camera_id column of its own — it's joined through
    ParkingSpot so this stays scoped the same way as every other query here.
    """
    return (
        Select(OccupancyState)
        .join(ParkingSpot, ParkingSpot.id == OccupancyState.spot_id)
        .where(ParkingSpot.camera_id == camera_id)
    )


def occupancy_events_for_camera(camera_id: int) -> Select[tuple[OccupancyEvent]]:
    """Select statement for the occupancy history of one camera's spots."""
    return (
        Select(OccupancyEvent)
        .join(ParkingSpot, ParkingSpot.id == OccupancyEvent.spot_id)
        .where(ParkingSpot.camera_id == camera_id)
    )


def camera_by_id(camera_id: int) -> Select[tuple[Camera]]:
    """Select statement for a single camera by id — the scope root itself."""
    return Select(Camera).where(Camera.id == camera_id)
