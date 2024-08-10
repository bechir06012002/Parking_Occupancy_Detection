from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, ParkingSpot


async def test_list_spots_404_for_unknown_camera(client: AsyncClient) -> None:
    response = await client.get("/api/cameras/999/spots")

    assert response.status_code == 404


async def test_list_spots_returns_camera_spots(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    camera = Camera(name="lot1", source_uri="x", resolution="1x1", sample_fps=1.0, is_active=True)
    db_session.add(camera)
    await db_session.flush()
    db_session.add(
        ParkingSpot(
            camera_id=camera.id,
            label="A1",
            polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            spot_index=0,
        )
    )
    await db_session.commit()

    response = await client.get(f"/api/cameras/{camera.id}/spots")

    assert response.status_code == 200
    [spot] = response.json()
    assert spot["label"] == "A1"
    assert spot["camera_id"] == camera.id
    assert spot["polygon"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


async def test_list_spots_empty_for_camera_with_no_spots(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    camera = Camera(name="lot1", source_uri="x", resolution="1x1", sample_fps=1.0, is_active=True)
    db_session.add(camera)
    await db_session.commit()

    response = await client.get(f"/api/cameras/{camera.id}/spots")

    assert response.status_code == 200
    assert response.json() == []
