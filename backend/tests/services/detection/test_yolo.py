from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from app.services.detection.yolo import YoloDetector
from app.services.frame import Frame


@dataclass
class FakeBoxes:
    xyxy: list[list[float]]
    conf: list[float]
    cls: list[float]


@dataclass
class FakeResult:
    names: dict[int, str]
    boxes: FakeBoxes | None = None
    obb: FakeBoxes | None = None


@dataclass
class FakeModel:
    results: list[FakeResult]
    calls: list[dict[str, object]] = field(default_factory=list)

    def predict(
        self, source: Sequence[Frame], *, conf: float, imgsz: int, verbose: bool
    ) -> list[FakeResult]:
        self.calls.append({"source": source, "conf": conf, "imgsz": imgsz, "verbose": verbose})
        return self.results


def _make_detector(
    model: FakeModel, confidence_threshold: float = 0.25, imgsz: int = 1024
) -> YoloDetector:
    # FakeModel is a structural stand-in for the private _YoloModelLike Protocol;
    # cast bridges the fixture's simpler types to the adapter's exact signature.
    return YoloDetector(
        weights_path="unused",
        confidence_threshold=confidence_threshold,
        imgsz=imgsz,
        model=cast(Any, model),
    )


COCO_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# DOTA vocabulary emitted by the aerial yolov8n-obb.pt checkpoint.
DOTA_NAMES = {0: "plane", 4: "small vehicle", 5: "large vehicle", 6: "ship"}


def _frame() -> Frame:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_filters_out_non_vehicle_classes() -> None:
    result = FakeResult(
        names=COCO_NAMES,
        boxes=FakeBoxes(
            xyxy=[[0, 0, 10, 10], [5, 5, 15, 15]],
            conf=[0.9, 0.8],
            cls=[0, 2],  # person, car
        ),
    )
    model = FakeModel(results=[result])
    detector = _make_detector(model)

    detections = detector.predict_batch([_frame()])

    assert len(detections) == 1
    [frame_detections] = detections
    assert [d.class_name for d in frame_detections] == ["car"]


def test_keeps_every_vehicle_class() -> None:
    result = FakeResult(
        names=COCO_NAMES,
        boxes=FakeBoxes(
            xyxy=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
            conf=[0.5, 0.5, 0.5, 0.5],
            cls=[2, 3, 5, 7],  # car, motorcycle, bus, truck
        ),
    )
    model = FakeModel(results=[result])
    detector = _make_detector(model)

    [frame_detections] = detector.predict_batch([_frame()])

    assert {d.class_name for d in frame_detections} == {"car", "motorcycle", "bus", "truck"}


def test_reads_obb_results_and_filters_to_dota_vehicle_classes() -> None:
    # OBB checkpoints leave `boxes` None and put the axis-aligned rect on
    # `obb.xyxy`; only small/large vehicle should survive the filter.
    result = FakeResult(
        names=DOTA_NAMES,
        obb=FakeBoxes(
            xyxy=[[10, 10, 30, 24], [40, 40, 90, 70], [0, 0, 5, 5]],
            conf=[0.7, 0.6, 0.9],
            cls=[4, 5, 6],  # small vehicle, large vehicle, ship
        ),
    )
    model = FakeModel(results=[result])
    detector = _make_detector(model)

    [frame_detections] = detector.predict_batch([_frame()])

    assert [d.class_name for d in frame_detections] == ["small vehicle", "large vehicle"]
    assert frame_detections[0].bbox == (10.0, 10.0, 30.0, 24.0)


def test_result_with_no_boxes_or_obb_yields_no_detections() -> None:
    result = FakeResult(names=COCO_NAMES, boxes=None, obb=None)
    model = FakeModel(results=[result])
    detector = _make_detector(model)

    [frame_detections] = detector.predict_batch([_frame()])

    assert frame_detections == []


def test_detection_bbox_and_confidence_pass_through() -> None:
    result = FakeResult(
        names=COCO_NAMES,
        boxes=FakeBoxes(xyxy=[[1.0, 2.0, 3.0, 4.0]], conf=[0.77], cls=[2]),
    )
    model = FakeModel(results=[result])
    detector = _make_detector(model)

    [[detection]] = detector.predict_batch([_frame()])

    assert detection.bbox == (1.0, 2.0, 3.0, 4.0)
    assert detection.confidence == 0.77
    assert detection.class_name == "car"


def test_predict_batch_passes_params_and_all_frames_in_one_call() -> None:
    model = FakeModel(results=[FakeResult(names=COCO_NAMES, boxes=None) for _ in range(3)])
    detector = _make_detector(model, confidence_threshold=0.42, imgsz=1280)

    frames = [_frame(), _frame(), _frame()]
    results = detector.predict_batch(frames)

    assert len(results) == 3
    assert len(model.calls) == 1  # one batched call, not one per frame
    assert model.calls[0]["conf"] == 0.42
    assert model.calls[0]["imgsz"] == 1280
    assert len(model.calls[0]["source"]) == 3  # type: ignore[arg-type]
