"""In-memory occupancy cache: a short-TTL read-through cache in front of
occupancy_state, so GET /api/occupancy doesn't hit the DB on every request.
The worker and API are separate processes, so nothing can push into this
cache directly — it refreshes from Postgres
itself once stale, rather than being kept live by the Observer hook in
services/occupancy/state.py (that hook only fires inside the worker).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import OccupancyState
from app.domain.occupancy import SpotState

DEFAULT_TTL_SECONDS = 1.0


@dataclass
class OccupancyCache:
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    clock: Callable[[], float] = time.monotonic
    _states: dict[int, SpotState] = field(default_factory=dict)
    _refreshed_at: float | None = None

    async def get_all(self, session: AsyncSession) -> list[SpotState]:
        now = self.clock()
        if self._refreshed_at is None or (now - self._refreshed_at) >= self.ttl_seconds:
            rows = (await session.execute(select(OccupancyState))).scalars().all()
            self._states = {
                row.spot_id: SpotState(
                    spot_id=row.spot_id,
                    is_occupied=row.is_occupied,
                    confidence=row.confidence,
                    track_id=row.track_id,
                )
                for row in rows
            }
            self._refreshed_at = now
        return list(self._states.values())
