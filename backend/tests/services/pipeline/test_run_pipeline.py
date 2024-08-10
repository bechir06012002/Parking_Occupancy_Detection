from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from app.domain.detection import Detection
from app.domain.occupancy import Spot, SpotState
from app.domain.pipeline import PipelineRunSummary
from app.domain.tracking import Track
from app.services.pipeline.run_pipeline import CameraRuntime, run_pipeline

SPOT = Spot(spot_id=1, polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))


class AlwaysReadyGrabber:
    def maybe_grab(self, *, now: float | None = None) -> str | None:
        return "frame"


class NeverReadyGrabber:
    def maybe_grab(self, *, now: float | None = None) -> str | None:
        return None


class RaisingGrabber:
    def maybe_grab(self, *, now: float | None = None) -> str | None:
        raise RuntimeError("camera exploded")


class NoopTracker:
    def update(self, detections: list[Detection]) -> list[Track]:
        return []


@dataclass
class FakeDetector:
    batch_sizes: list[int] = field(default_factory=list)

    def predict_batch(self, frames: Sequence[object]) -> list[list[Detection]]:
        self.batch_sizes.append(len(frames))
        return [[] for _ in frames]


@dataclass
class FakeStrategy:
    is_occupied: bool = True

    def match(self, tracks: Sequence[Track], spots: Sequence[Spot]) -> list[SpotState]:
        return [
            SpotState(
                spot_id=s.spot_id,
                is_occupied=self.is_occupied,
                confidence=0.9,
                track_id=1 if self.is_occupied else None,
            )
            for s in spots
        ]


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


async def _fake_sleep(seconds: float) -> None:
    return None


def _camera(camera_id: int, grabber: object) -> CameraRuntime:
    # The Fake*Grabber classes are structural stand-ins for the private
    # _GrabberLike Protocol; cast bridges the fixture's loose typing.
    return CameraRuntime(
        camera_id=camera_id,
        spots=[SPOT],
        grabber=cast(Any, grabber),
        tracker=NoopTracker(),
    )


async def test_persists_each_tick_with_correct_transition_flag() -> None:
    persisted: list[tuple[SpotState, bool]] = []
    finished: list[PipelineRunSummary] = []

    async def persist(state: SpotState, transitioned: bool) -> None:
        persisted.append((state, transitioned))

    async def on_run_finished(summary: PipelineRunSummary) -> None:
        finished.append(summary)

    await run_pipeline(
        cameras=[_camera(1, AlwaysReadyGrabber())],
        detector=FakeDetector(),
        strategy=FakeStrategy(is_occupied=True),
        smoothing_window=5,
        persist=persist,
        model_version="yolov8n",
        tick_interval_seconds=0.0,
        on_run_finished=on_run_finished,
        max_ticks=3,
        clock=FakeClock(),
        sleep=_fake_sleep,
    )

    assert [transitioned for _, transitioned in persisted] == [True, False, False]
    assert len(finished) == 1
    assert finished[0].camera_id == 1
    assert finished[0].frames_processed == 3
    assert finished[0].avg_latency_ms is not None
    assert finished[0].avg_latency_ms > 0
    assert finished[0].model_version == "yolov8n"


async def test_multiple_ready_cameras_share_one_batched_predict_call() -> None:
    detector = FakeDetector()

    async def persist(state: SpotState, transitioned: bool) -> None:
        pass

    await run_pipeline(
        cameras=[_camera(1, AlwaysReadyGrabber()), _camera(2, AlwaysReadyGrabber())],
        detector=detector,
        strategy=FakeStrategy(),
        smoothing_window=5,
        persist=persist,
        model_version="yolov8n",
        tick_interval_seconds=0.0,
        max_ticks=2,
        clock=FakeClock(),
        sleep=_fake_sleep,
    )

    assert detector.batch_sizes == [2, 2]  # one call per tick, covering both cameras


async def test_camera_never_ready_never_triggers_inference() -> None:
    detector = FakeDetector()
    finished: list[PipelineRunSummary] = []

    async def persist(state: SpotState, transitioned: bool) -> None:
        raise AssertionError("should never be called")

    async def on_run_finished(summary: PipelineRunSummary) -> None:
        finished.append(summary)

    await run_pipeline(
        cameras=[_camera(1, NeverReadyGrabber())],
        detector=detector,
        strategy=FakeStrategy(),
        smoothing_window=5,
        persist=persist,
        model_version="yolov8n",
        tick_interval_seconds=0.0,
        on_run_finished=on_run_finished,
        max_ticks=3,
        clock=FakeClock(),
        sleep=_fake_sleep,
    )

    assert detector.batch_sizes == []
    assert finished[0].frames_processed == 0
    assert finished[0].avg_latency_ms is None
    assert finished[0].p95_latency_ms is None


async def test_one_bad_camera_does_not_stop_the_others() -> None:
    persisted_camera_ids: list[int] = []

    async def persist(state: SpotState, transitioned: bool) -> None:
        persisted_camera_ids.append(state.spot_id)

    await run_pipeline(
        cameras=[_camera(1, RaisingGrabber()), _camera(2, AlwaysReadyGrabber())],
        detector=FakeDetector(),
        strategy=FakeStrategy(),
        smoothing_window=5,
        persist=persist,
        model_version="yolov8n",
        tick_interval_seconds=0.0,
        max_ticks=2,
        clock=FakeClock(),
        sleep=_fake_sleep,
    )

    assert persisted_camera_ids  # camera 2 kept producing observations


async def test_on_run_finished_still_called_when_persist_raises() -> None:
    finished: list[PipelineRunSummary] = []

    async def flaky_persist(state: SpotState, transitioned: bool) -> None:
        raise RuntimeError("db is down")

    async def on_run_finished(summary: PipelineRunSummary) -> None:
        finished.append(summary)

    with pytest.raises(RuntimeError, match="db is down"):
        await run_pipeline(
            cameras=[_camera(1, AlwaysReadyGrabber())],
            detector=FakeDetector(),
            strategy=FakeStrategy(),
            smoothing_window=5,
            persist=flaky_persist,
            model_version="yolov8n",
            tick_interval_seconds=0.0,
            on_run_finished=on_run_finished,
            max_ticks=3,
            clock=FakeClock(),
            sleep=_fake_sleep,
        )

    assert len(finished) == 1
