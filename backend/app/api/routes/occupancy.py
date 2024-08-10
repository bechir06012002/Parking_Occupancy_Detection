from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import DbSession, OccupancyCacheDep
from app.db.models import OccupancyEvent, ParkingSpot

router = APIRouter(prefix="/api/occupancy", tags=["occupancy"])


class OccupancyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    spot_id: int
    is_occupied: bool
    confidence: float
    track_id: int | None


class OccupancyEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    spot_id: int
    is_occupied: bool
    confidence: float
    occurred_at: datetime


@router.get("", response_model=list[OccupancyOut])
async def get_occupancy(session: DbSession, cache: OccupancyCacheDep) -> list[OccupancyOut]:
    states = await cache.get_all(session)
    return [OccupancyOut.model_validate(state) for state in states]


@router.get("/{spot_id}/history", response_model=list[OccupancyEventOut])
async def get_occupancy_history(
    spot_id: int,
    session: DbSession,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[OccupancyEventOut]:
    spot_exists = (
        await session.execute(select(ParkingSpot.id).where(ParkingSpot.id == spot_id))
    ).scalar_one_or_none()
    if spot_exists is None:
        raise HTTPException(status_code=404, detail="spot not found")

    rows = (
        (
            await session.execute(
                select(OccupancyEvent)
                .where(OccupancyEvent.spot_id == spot_id)
                .order_by(OccupancyEvent.occurred_at.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [OccupancyEventOut.model_validate(row) for row in rows]
