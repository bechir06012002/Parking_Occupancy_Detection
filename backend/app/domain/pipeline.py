from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PipelineRunSummary:
    """One completed (or interrupted) run of the pipeline against one camera.

    avg/p95 latency are computed from real per-frame capture-to-persist
    timings and are None only when zero frames were processed during the
    run.
    """

    camera_id: int
    started_at: datetime
    ended_at: datetime
    frames_processed: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    model_version: str
