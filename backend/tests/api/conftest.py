"""API test fixtures: a disposable Postgres schema per test.

Ruled out `testcontainers` in favor of either SQLite or a disposable
Postgres schema. SQLite is a poor fit here — parking_spots.polygon
is JSONB and app/db/writer.py uses Postgres's INSERT...ON CONFLICT — so this
uses a real, uniquely-named schema on the already-running local/CI Postgres
via SQLAlchemy's `schema_translate_map`.

The schema (and its engine) is created fresh per test function rather than
shared across a module: pytest-asyncio gives each test its own event loop
by default, and an asyncpg connection pool built in one loop breaks with
"another operation is in progress" if reused from another — this sidesteps
that entirely instead of fighting pytest-asyncio's loop-scope config.
"""

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.cache import OccupancyCache
from app.api.deps import get_occupancy_cache
from app.core.config import get_settings
from app.db.models import Base
from app.db.session import get_session
from app.main import app


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    database_url = get_settings().database_url
    schema = f"test_{uuid4().hex[:8]}"

    admin_engine = create_async_engine(database_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    engine = create_async_engine(database_url).execution_options(
        schema_translate_map={None: schema}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()
    async with admin_engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
    await admin_engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_occupancy_cache] = OccupancyCache
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
