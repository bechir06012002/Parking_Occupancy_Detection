"""YOLOv8 detection adapter (Adapter pattern).

Isolates the Ultralytics call boundary behind `YoloDetector.predict_batch`.
Weights path, confidence threshold, and inference image size are constructor
parameters, never literals in this module — the caller (the worker, in P4)
reads them from `Settings` and passes them in. The model is expected to be
loaded once and reused for every scheduler tick across every camera (one
`predict_batch` call per tick), not re-loaded per frame.

Two result shapes are handled transparently:

- axis-aligned detection models (stock COCO `yolov8n.pt`) put boxes on
  `result.boxes`;
- oriented-bbox models (the DOTA-pretrained `yolov8n-obb.pt` aerial default,
  see docs/evaluation.md) put them on `result.obb`, where `.xyxy` is the
  axis-aligned bounding rect of each oriented box.

Both expose `.xyxy` / `.conf` / `.cls`, so downstream only ever sees a plain
axis-aligned `Detection`. The vehicle-class filter lives here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

from ultralytics.models.yolo.model import YOLO

from app.domain.detection import VEHICLE_CLASSES, Detection
from app.services.frame import Frame


class _YoloBoxesLike(Protocol):
    xyxy: Iterable[Iterable[float]]
    conf: Iterable[float]
    cls: Iterable[float]


class _YoloResultLike(Protocol):
    names: Mapping[int, str]
    boxes: _YoloBoxesLike | None
    obb: _YoloBoxesLike | None


class _YoloModelLike(Protocol):
    def predict(
        self, source: Sequence[Frame], *, conf: float, imgsz: int, verbose: bool
    ) -> Sequence[_YoloResultLike]: ...


def _detections_from_result(result: _YoloResultLike) -> list[Detection]:
    # OBB checkpoints populate `.obb` and leave `.boxes` None; detection
    # checkpoints do the reverse.
    boxes = result.obb if result.obb is not None else result.boxes
    if boxes is None:
        return []
    detections: list[Detection] = []
    for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls, strict=True):
        class_name = result.names[int(cls)]
        if class_name not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        detections.append(
            Detection(bbox=(x1, y1, x2, y2), confidence=float(conf), class_name=class_name)
        )
    return detections


class YoloDetector:
    """Loads a YOLO model once; batches frames from every camera into one call."""

    def __init__(
        self,
        *,
        weights_path: str,
        confidence_threshold: float,
        imgsz: int,
        model: _YoloModelLike | None = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._imgsz = imgsz
        self._model: _YoloModelLike = (
            model if model is not None else cast(_YoloModelLike, YOLO(weights_path))
        )

    def predict_batch(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Run one batched inference call and return vehicle-only detections per frame."""
        results = self._model.predict(
            frames, conf=self._confidence_threshold, imgsz=self._imgsz, verbose=False
        )
        return [_detections_from_result(result) for result in results]
