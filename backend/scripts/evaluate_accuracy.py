"""Accuracy evaluation: predicted occupancy vs hand-labeled ground truth.

Runs the exact same detect -> track -> match stages run_pipeline.py uses
(YoloDetector, ByteTrackAdapter, CentroidCoverageStrategy — config-driven
from Settings, same as production) against one frame of a real evaluation video,
compares each spot's predicted is_occupied against a hand-labeled
ground-truth file, and reports per-spot + aggregate accuracy,
precision/recall/F1 on the "occupied" class, and a confusion matrix.

This is a standalone script, not a pytest test — it needs a real video and
real model weights, neither of which belong in CI.

Usage:
    uv run python scripts/evaluate_accuracy.py \
        --video ../data/parking_lot.mov \
        --ground-truth scripts/ground_truth/parking_lot.json \
        --output scripts/ground_truth/parking_lot_results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from cv2.typing import MatLike

from app.core.config import get_settings
from app.domain.occupancy import Spot
from app.services.detection.yolo import YoloDetector
from app.services.occupancy.matching import CentroidCoverageStrategy
from app.services.tracking.byte_track import ByteTrackAdapter


def load_ground_truth(path: Path) -> tuple[list[Spot], dict[int, bool], dict[int, str]]:
    data = json.loads(path.read_text())
    spots: list[Spot] = []
    truth: dict[int, bool] = {}
    labels: dict[int, str] = {}
    for i, entry in enumerate(data["spots"]):
        spots.append(Spot(spot_id=i, polygon=tuple((p[0], p[1]) for p in entry["polygon"])))
        truth[i] = entry["is_occupied"]
        labels[i] = entry["label"]
    return spots, truth, labels


def load_frame(video_path: Path, frame_index: int) -> MatLike:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
        return frame
    finally:
        capture.release()


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    gt_data = json.loads(args.ground_truth.read_text())
    spots, truth, labels = load_ground_truth(args.ground_truth)
    frame = load_frame(args.video, gt_data["frame_index"])

    settings = get_settings()
    detector = YoloDetector(
        weights_path=settings.yolo_weights_path,
        confidence_threshold=settings.yolo_conf_threshold,
        imgsz=settings.yolo_imgsz,
    )
    tracker = ByteTrackAdapter()
    strategy = CentroidCoverageStrategy(coverage_threshold=settings.occupancy_coverage_threshold)

    [detections] = detector.predict_batch([frame])
    tracks = tracker.update(detections)
    predictions = strategy.match(tracks, spots)
    predicted_by_spot = {p.spot_id: p.is_occupied for p in predictions}

    tp = fp = tn = fn = 0
    rows: list[tuple[str, bool, bool, bool]] = []
    for spot in spots:
        actual = truth[spot.spot_id]
        predicted = predicted_by_spot[spot.spot_id]
        correct = actual == predicted
        rows.append((labels[spot.spot_id], actual, predicted, correct))
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1

    total = len(spots)
    correct_count = sum(1 for *_, correct in rows if correct)
    accuracy = correct_count / total
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0
        else float("nan")
    )

    print(f"{'label':<6} {'actual':<9} {'predicted':<10} {'correct':<8}")
    for label, actual, predicted, correct in rows:
        print(f"{label:<6} {str(actual):<9} {str(predicted):<10} {str(correct):<8}")
    print()
    print(f"video: {args.video}  frame_index: {gt_data['frame_index']}")
    print(
        f"model: {settings.yolo_weights_path} "
        f"(conf={settings.yolo_conf_threshold}, imgsz={settings.yolo_imgsz}, "
        f"coverage_threshold={settings.occupancy_coverage_threshold})"
    )
    print(f"spots evaluated: {total}")
    print(f"aggregate accuracy: {_fmt(accuracy)} ({correct_count}/{total})")
    print(f"confusion matrix (positive class = occupied): TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"precision (occupied): {_fmt(precision)}")
    print(f"recall (occupied): {_fmt(recall)}")
    print(f"f1 (occupied): {_fmt(f1)}")

    if args.output is not None:
        args.output.write_text(
            json.dumps(
                {
                    "video": str(args.video),
                    "frame_index": gt_data["frame_index"],
                    "model_version": settings.yolo_weights_path,
                    "yolo_conf_threshold": settings.yolo_conf_threshold,
                    "yolo_imgsz": settings.yolo_imgsz,
                    "occupancy_coverage_threshold": settings.occupancy_coverage_threshold,
                    "spots_evaluated": total,
                    "aggregate_accuracy": accuracy,
                    "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                    "precision_occupied": None if math.isnan(precision) else precision,
                    "recall_occupied": None if math.isnan(recall) else recall,
                    "f1_occupied": None if math.isnan(f1) else f1,
                    "per_spot": [
                        {
                            "label": label,
                            "actual": actual,
                            "predicted": predicted,
                            "correct": correct,
                        }
                        for label, actual, predicted, correct in rows
                    ],
                },
                indent=2,
            )
        )
        print(f"\nwrote results to {args.output}")


if __name__ == "__main__":
    main()
