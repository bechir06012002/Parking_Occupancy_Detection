from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, OccupancyState, ParkingSpot, PipelineRun


async def test_metrics_all_zero_when_empty(client: AsyncClient) -> None:
    response = await client.get("/api/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["total_cameras"] == 0
    assert body["total_spots"] == 0
    assert body["occupied_spots"] == 0
    assert body["free_spots"] == 0
    assert body["latest_runs"] == []


async def test_metrics_reflect_seeded_data(client: AsyncClient, db_session: AsyncSession) -> None:
    camera = Camera(name="lot1", source_uri="x", resolution="1x1", sample_fps=1.0, is_active=True)
    db_session.add(camera)
    await db_session.flush()

    occupied_spot = ParkingSpot(camera_id=camera.id, label="A1", polygon=[[0, 0]], spot_index=0)
    free_spot = ParkingSpot(camera_id=camera.id, label="A2", polygon=[[0, 0]], spot_index=1)
    db_session.add_all([occupied_spot, free_spot])
    await db_session.flush()

    now = datetime.now(UTC)
    db_session.add_all(
        [
            OccupancyState(
                spot_id=occupied_spot.id,
                is_occupied=True,
                confidence=0.9,
                track_id=1,
                last_changed_at=now,
                last_seen_at=now,
            ),
            OccupancyState(
                spot_id=free_spot.id,
                is_occupied=False,
                confidence=0.0,
                track_id=None,
                last_changed_at=now,
                last_seen_at=now,
            ),
            PipelineRun(
                camera_id=camera.id,
                started_at=now,
                ended_at=now,
                frames_processed=10,
                avg_latency_ms=123.4,
                p95_latency_ms=456.7,
                model_version="yolov8n.pt",
            ),
        ]
    )
    await db_session.commit()

    response = await client.get("/api/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["total_cameras"] == 1
    assert body["total_spots"] == 2
    assert body["occupied_spots"] == 1
    assert body["free_spots"] == 1
    [run] = body["latest_runs"]
    assert run["camera_id"] == camera.id
    assert run["frames_processed"] == 10
    assert run["avg_latency_ms"] == 123.4
