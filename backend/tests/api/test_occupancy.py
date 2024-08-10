from datetime import UTC, datetime, timedelta

from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.cache import OccupancyCache
from app.api.deps import get_occupancy_cache
from app.db.models import Camera, OccupancyEvent, OccupancyState, ParkingSpot
from app.db.session import get_session
from app.main import app


async def _seed_camera_and_spot(db_session: AsyncSession) -> tuple[Camera, ParkingSpot]:
    camera = Camera(name="lot1", source_uri="x", resolution="1x1", sample_fps=1.0, is_active=True)
    db_session.add(camera)
    await db_session.flush()
    spot = ParkingSpot(
        camera_id=camera.id,
        label="A1",
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        spot_index=0,
    )
    db_session.add(spot)
    await db_session.flush()
    return camera, spot


async def test_get_occupancy_empty(client: AsyncClient) -> None:
    response = await client.get("/api/occupancy")

    assert response.status_code == 200
    assert response.json() == []


async def test_get_occupancy_reflects_state(client: AsyncClient, db_session: AsyncSession) -> None:
    _, spot = await _seed_camera_and_spot(db_session)
    now = datetime.now(UTC)
    db_session.add(
        OccupancyState(
            spot_id=spot.id,
            is_occupied=True,
            confidence=0.9,
            track_id=5,
            last_changed_at=now,
            last_seen_at=now,
        )
    )
    await db_session.commit()

    response = await client.get("/api/occupancy")

    assert response.status_code == 200
    [state] = response.json()
    assert state["spot_id"] == spot.id
    assert state["is_occupied"] is True
    assert state["confidence"] == 0.9
    assert state["track_id"] == 5


async def test_occupancy_history_404_for_unknown_spot(client: AsyncClient) -> None:
    response = await client.get("/api/occupancy/999/history")

    assert response.status_code == 404


async def test_occupancy_history_returns_events_newest_first(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    _, spot = await _seed_camera_and_spot(db_session)
    older = datetime.now(UTC) - timedelta(minutes=5)
    newer = datetime.now(UTC)
    db_session.add_all(
        [
            OccupancyEvent(spot_id=spot.id, is_occupied=False, confidence=0.0, occurred_at=older),
            OccupancyEvent(spot_id=spot.id, is_occupied=True, confidence=0.9, occurred_at=newer),
        ]
    )
    await db_session.commit()

    response = await client.get(f"/api/occupancy/{spot.id}/history")

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 2
    assert events[0]["is_occupied"] is True  # newest first
    assert events[1]["is_occupied"] is False


async def test_occupancy_cache_serves_stale_within_ttl_then_refreshes(
    db_session: AsyncSession,
) -> None:
    _, spot = await _seed_camera_and_spot(db_session)
    now = datetime.now(UTC)
    db_session.add(
        OccupancyState(
            spot_id=spot.id,
            is_occupied=False,
            confidence=0.0,
            track_id=None,
            last_changed_at=now,
            last_seen_at=now,
        )
    )
    await db_session.commit()

    fake_time = {"value": 0.0}
    cache = OccupancyCache(ttl_seconds=1.0, clock=lambda: fake_time["value"])

    async def override_get_session() -> object:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_occupancy_cache] = lambda: cache
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            first = await ac.get("/api/occupancy")
            assert first.json()[0]["is_occupied"] is False

            # Change the DB directly, bypassing the cache.
            await db_session.execute(
                update(OccupancyState)
                .where(OccupancyState.spot_id == spot.id)
                .values(is_occupied=True)
            )
            await db_session.commit()

            fake_time["value"] += 0.5  # still within the TTL window
            still_cached = await ac.get("/api/occupancy")
            assert still_cached.json()[0]["is_occupied"] is False

            fake_time["value"] += 1.0  # now past the TTL
            refreshed = await ac.get("/api/occupancy")
            assert refreshed.json()[0]["is_occupied"] is True
    finally:
        app.dependency_overrides.clear()
