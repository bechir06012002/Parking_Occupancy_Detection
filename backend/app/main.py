from fastapi import FastAPI

from app.api.routes import cameras, health, metrics, occupancy, spots

app = FastAPI(title="Smart Parking Occupancy Detector")

app.include_router(health.router)
app.include_router(cameras.router)
app.include_router(spots.router)
app.include_router(occupancy.router)
app.include_router(metrics.router)
