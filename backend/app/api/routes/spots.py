from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import DbSession
from app.db.models import Camera
from app.db.scoped import spots_for_camera

router = APIRouter(prefix="/api/cameras", tags=["spots"])


class SpotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    label: str
    polygon: list[list[float]]
    spot_index: int


@router.get("/{camera_id}/spots", response_model=list[SpotOut])
async def list_spots(camera_id: int, session: DbSession) -> list[SpotOut]:
    camera = (
        await session.execute(select(Camera).where(Camera.id == camera_id))
    ).scalar_one_or_none()
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")

    rows = (await session.execute(spots_for_camera(camera_id))).scalars().all()
    return [SpotOut.model_validate(row) for row in rows]
