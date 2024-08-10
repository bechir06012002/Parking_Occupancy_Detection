from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.api.deps import DbSession
from app.db.models import Camera

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # source_uri is deliberately excluded — RTSP URLs can carry embedded
    # credentials; no reason to put them on the wire either, even to this
    # system's own trusted caller.
    id: int
    name: str
    resolution: str
    sample_fps: float
    is_active: bool


@router.get("", response_model=list[CameraOut])
async def list_cameras(session: DbSession) -> list[CameraOut]:
    rows = (await session.execute(select(Camera).order_by(Camera.id))).scalars().all()
    return [CameraOut.model_validate(row) for row in rows]
