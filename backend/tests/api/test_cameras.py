from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera


async def test_list_cameras_empty(client: AsyncClient) -> None:
    response = await client.get("/api/cameras")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_cameras_returns_seeded_camera_without_source_uri(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(
        Camera(
            name="lot1",
            source_uri="rtsp://user:pass@host/stream",
            resolution="640x480",
            sample_fps=1.0,
            is_active=True,
        )
    )
    await db_session.commit()

    response = await client.get("/api/cameras")

    assert response.status_code == 200
    [camera] = response.json()
    assert camera["name"] == "lot1"
    assert camera["resolution"] == "640x480"
    assert camera["is_active"] is True
    assert "source_uri" not in camera
