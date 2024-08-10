from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    yolo_weights_path: str
    yolo_conf_threshold: float
    # Inference image size passed to `model.predict(imgsz=...)`. The aerial
    # (DOTA-pretrained OBB) checkpoint needs a larger size than YOLO's 640
    # default — a car in this 1920px-wide top-down footage is ~65px and
    # shrinks below the detection floor at 640. Config-driven, never a
    # literal in `services/detection/`.
    yolo_imgsz: int
    # CentroidCoverageStrategy's gate: minimum fraction of a detected vehicle
    # box that must lie inside a spot for the spot to count as occupied.
    # The default matching strategy.
    occupancy_coverage_threshold: float
    # CentroidIoUStrategy's gate — kept for the swappable alternative strategy.
    occupancy_iou_threshold: float
    smoothing_window: int
    # Batch-scheduler tick interval (1/sample_fps seconds), not a per-camera
    # rate — each camera paces its own grabs from its own `cameras.sample_fps`
    # DB column. This should be >= the fastest camera's rate to sample it
    # accurately; see worker/run_worker.py.
    sample_fps: float


@lru_cache
def get_settings() -> Settings:
    return Settings()
