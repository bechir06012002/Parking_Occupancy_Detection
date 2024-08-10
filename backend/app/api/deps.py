"""DI: DB session, settings, occupancy cache reader."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.cache import OccupancyCache
from app.core.config import Settings, get_settings
from app.db.session import get_session

_occupancy_cache = OccupancyCache()


def get_occupancy_cache() -> OccupancyCache:
    return _occupancy_cache


DbSession = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
OccupancyCacheDep = Annotated[OccupancyCache, Depends(get_occupancy_cache)]
