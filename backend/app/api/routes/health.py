from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.api.deps import DbSession

router = APIRouter(tags=["health"])


class HealthOut(BaseModel):
    status: str


@router.get("/healthz", response_model=HealthOut)
async def healthz(session: DbSession, response: Response) -> HealthOut:
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthOut(status="db unreachable")
    return HealthOut(status="ok")
