from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.api.deps import DbSession
from app.db.models import Camera, OccupancyState, ParkingSpot, PipelineRun

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


class LatestRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    camera_id: int
    started_at: datetime
    ended_at: datetime | None
    frames_processed: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    model_version: str


class MetricsOut(BaseModel):
    total_cameras: int
    total_spots: int
    occupied_spots: int
    free_spots: int
    latest_runs: list[LatestRunOut]


@router.get("", response_model=MetricsOut)
async def get_metrics(session: DbSession) -> MetricsOut:
    total_cameras = (await session.execute(select(func.count()).select_from(Camera))).scalar_one()
    total_spots = (
        await session.execute(select(func.count()).select_from(ParkingSpot))
    ).scalar_one()
    occupied_spots = (
        await session.execute(
            select(func.count())
            .select_from(OccupancyState)
            .where(OccupancyState.is_occupied.is_(True))
        )
    ).scalar_one()

    runs = (
        (await session.execute(select(PipelineRun).order_by(PipelineRun.started_at.desc())))
        .scalars()
        .all()
    )
    latest_by_camera: dict[int, PipelineRun] = {}
    for run in runs:
        latest_by_camera.setdefault(run.camera_id, run)

    return MetricsOut(
        total_cameras=total_cameras,
        total_spots=total_spots,
        occupied_spots=occupied_spots,
        free_spots=total_spots - occupied_spots,
        latest_runs=[LatestRunOut.model_validate(run) for run in latest_by_camera.values()],
    )
