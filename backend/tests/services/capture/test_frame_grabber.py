from typing import Any, cast

from app.services.capture.frame_grabber import FrameGrabber


class FakeCapture:
    def __init__(self, results: list[tuple[bool, object]]) -> None:
        self._results = list(results)
        self.read_calls = 0
        self.released = False

    def read(self) -> tuple[bool, object]:
        self.read_calls += 1
        return self._results.pop(0)

    def release(self) -> None:
        self.released = True


class RaisingCapture:
    def read(self) -> tuple[bool, object]:
        raise RuntimeError("camera disconnected")

    def release(self) -> None:
        pass


def _grabber(capture: object, *, sample_interval_seconds: float = 1.0) -> FrameGrabber:
    # FakeCapture/RaisingCapture are structural stand-ins for the private
    # _CaptureLike Protocol; cast bridges the fixture's loose typing.
    return FrameGrabber(
        camera_id=1,
        source_uri="unused",
        sample_interval_seconds=sample_interval_seconds,
        capture_factory=cast(Any, lambda: capture),
    )


def test_first_call_always_grabs() -> None:
    capture = FakeCapture([(True, "frame-0")])
    grabber = _grabber(capture)

    frame = grabber.maybe_grab(now=0.0)

    assert frame == "frame-0"
    assert capture.read_calls == 1


def test_second_call_too_soon_returns_none_without_reading() -> None:
    capture = FakeCapture([(True, "frame-0")])
    grabber = _grabber(capture, sample_interval_seconds=1.0)

    grabber.maybe_grab(now=0.0)
    frame = grabber.maybe_grab(now=0.5)

    assert frame is None
    assert capture.read_calls == 1


def test_call_after_interval_elapses_grabs_again() -> None:
    capture = FakeCapture([(True, "frame-0"), (True, "frame-1")])
    grabber = _grabber(capture, sample_interval_seconds=1.0)

    grabber.maybe_grab(now=0.0)
    frame = grabber.maybe_grab(now=1.5)

    assert frame == "frame-1"
    assert capture.read_calls == 2


def test_failed_read_returns_none() -> None:
    capture = FakeCapture([(False, None)])
    grabber = _grabber(capture)

    frame = grabber.maybe_grab(now=0.0)

    assert frame is None


def test_exception_during_read_is_caught_and_returns_none() -> None:
    grabber = _grabber(RaisingCapture())

    frame = grabber.maybe_grab(now=0.0)

    assert frame is None


def test_capture_is_opened_lazily_and_only_once() -> None:
    capture = FakeCapture([(True, "frame-0"), (True, "frame-1")])
    open_count = 0

    def factory() -> FakeCapture:
        nonlocal open_count
        open_count += 1
        return capture

    grabber = FrameGrabber(
        camera_id=1,
        source_uri="unused",
        sample_interval_seconds=1.0,
        capture_factory=cast(Any, factory),
    )
    assert open_count == 0  # not opened at construction

    grabber.maybe_grab(now=0.0)
    grabber.maybe_grab(now=1.0)

    assert open_count == 1  # opened once, reused across calls


def test_release_releases_underlying_capture() -> None:
    capture = FakeCapture([(True, "frame-0")])
    grabber = _grabber(capture)
    grabber.maybe_grab(now=0.0)

    grabber.release()

    assert capture.released is True
