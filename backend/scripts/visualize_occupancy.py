"""Visual demo: draws predicted occupancy directly onto the video.

Runs the real detect -> track -> match chain (YoloDetector, ByteTrackAdapter,
CentroidCoverageStrategy — same components run_pipeline.py uses, config-driven
from Settings) against each processed frame of a real video, draws every
spot's polygon in green (predicted free) or red (predicted occupied) plus
the raw vehicle detections in yellow and a running header, and writes the
result to an output video file so a human can watch it rather than read
JSON.

By default it processes the whole video (`--max-frames 0`), so one command
produces a full-length annotated clip.

This is a demo/dev tool, not part of the formal evaluation — for the
accuracy numbers see scripts/evaluate_accuracy.py and docs/evaluation.md.

Usage:
    uv run python scripts/visualize_occupancy.py \
        --video ../data/parking_lot.mov \
        --spots-json scripts/ground_truth/parking_lot.json \
        --output ../data/parking_lot_annotated.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.domain.occupancy import Spot
from app.services.detection.yolo import YoloDetector
from app.services.occupancy.matching import CentroidCoverageStrategy
from app.services.tracking.byte_track import ByteTrackAdapter

FREE_COLOR = (0, 200, 0)  # green, BGR
OCCUPIED_COLOR = (0, 0, 220)  # red, BGR
DETECTION_COLOR = (0, 200, 220)  # amber, BGR
HUD_ACCENT = (200, 140, 0)  # teal-blue accent bar, BGR
HUD_BG = (30, 24, 18)  # near-black panel background, BGR


def _draw_occupancy_hud(frame: np.ndarray, *, occupied: int, total: int, width: int) -> None:
    """Draw a bold "OCCUPIED n/total" status card in the top-right corner.

    A semi-transparent panel with an accent bar and two-line text, distinct
    from the small top-left telemetry line — meant to be readable from
    across a room, the number a viewer actually cares about.
    """
    free = total - occupied
    count_text = f"{occupied}/{total}"
    label_text = "SPOTS OCCUPIED"
    sub_text = f"{free} FREE"

    count_scale, count_thickness = 1.1, 3
    label_scale, sub_scale, small_thickness = 0.5, 0.5, 1

    (count_w, count_h), _ = cv2.getTextSize(
        count_text, cv2.FONT_HERSHEY_DUPLEX, count_scale, count_thickness
    )
    (label_w, _), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, label_scale, small_thickness
    )
    (sub_w, _), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, small_thickness)

    pad_x, pad_top, pad_mid, pad_bottom = 18, 14, 6, 12
    panel_w = max(count_w, label_w, sub_w) + 2 * pad_x
    panel_h = pad_top + 16 + pad_mid + count_h + pad_mid + 14 + pad_bottom
    margin = 14
    x2, y1 = width - margin, margin
    x1 = x2 - panel_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y1 + panel_h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y1 + 4), HUD_ACCENT, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y1 + panel_h), HUD_ACCENT, 1)

    cv2.putText(
        frame,
        label_text,
        (x2 - pad_x - label_w, y1 + pad_top + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (170, 170, 170),
        small_thickness,
        cv2.LINE_AA,
    )
    count_y = y1 + pad_top + 16 + pad_mid + count_h
    cv2.putText(
        frame,
        count_text,
        (x2 - pad_x - count_w, count_y),
        cv2.FONT_HERSHEY_DUPLEX,
        count_scale,
        (255, 255, 255),
        count_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        sub_text,
        (x2 - pad_x - sub_w, count_y + pad_mid + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_scale,
        FREE_COLOR,
        small_thickness,
        cv2.LINE_AA,
    )


def load_spots(path: Path) -> list[Spot]:
    data = json.loads(path.read_text())
    return [
        Spot(spot_id=i, polygon=tuple((p[0], p[1]) for p in entry["polygon"]))
        for i, entry in enumerate(data["spots"])
    ]


def load_labels(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text())
    return {i: entry["label"] for i, entry in enumerate(data["spots"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--spots-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video (default)")
    parser.add_argument("--every-n-frames", type=int, default=1, help="Sample every Nth frame")
    parser.add_argument("--no-detections", action="store_true", help="Don't draw raw vehicle boxes")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Output scale factor (e.g. 0.5 for a smaller file)"
    )
    parser.add_argument(
        "--label-limit",
        type=int,
        default=40,
        help="Draw per-spot text labels only up to this many spots (else outlines only)",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=0,
        help="Spot-outline thickness in px (0 = auto: 3 up to 300 spots, 2 above)",
    )
    args = parser.parse_args()

    settings = get_settings()
    spots = load_spots(args.spots_json)
    labels = load_labels(args.spots_json)

    detector = YoloDetector(
        weights_path=settings.yolo_weights_path,
        confidence_threshold=settings.yolo_conf_threshold,
        imgsz=settings.yolo_imgsz,
    )
    tracker = ByteTrackAdapter()
    strategy = CentroidCoverageStrategy(coverage_threshold=settings.occupancy_coverage_threshold)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_w, out_h = int(width * args.scale), int(height * args.scale)

    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter.fourcc(*"mp4v"), fps, (out_w, out_h))

    frame_index = 0
    written = 0
    try:
        while args.max_frames == 0 or written < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % args.every_n_frames != 0:
                continue

            [detections] = detector.predict_batch([frame])
            tracks = tracker.update(detections)
            predictions = {p.spot_id: p for p in strategy.match(tracks, spots)}

            annotated = frame.copy()

            if not args.no_detections:
                for det in detections:
                    x1, y1, x2, y2 = (int(v) for v in det.bbox)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), DETECTION_COLOR, 1)

            # Past a few dozen spots, one text label per polygon turns into
            # unreadable clutter — the HUD's live counts carry that
            # information instead, and the outline color is what's meant to
            # read from a distance for a full-lot layout.
            draw_labels = len(spots) <= args.label_limit
            line_thickness = args.line_thickness or (3 if len(spots) <= 300 else 2)

            occupied = 0
            for spot in spots:
                prediction = predictions[spot.spot_id]
                occupied += int(prediction.is_occupied)
                color = OCCUPIED_COLOR if prediction.is_occupied else FREE_COLOR
                points = [(int(x), int(y)) for x, y in spot.polygon]
                points_np = np.array(points, dtype=np.int32)
                cv2.polylines(annotated, [points_np], True, color, line_thickness)
                if draw_labels:
                    text = f"{labels[spot.spot_id]}:{'OCC' if prediction.is_occupied else 'free'}"
                    cv2.putText(
                        annotated,
                        text,
                        (points[0][0], max(0, points[0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        2,
                    )

            header = (
                f"vehicles detected: {len(detections):3d}   "
                f"model: {Path(settings.yolo_weights_path).name}"
            )
            cv2.rectangle(annotated, (0, 0), (width, 34), (0, 0, 0), -1)
            cv2.putText(
                annotated, header, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            _draw_occupancy_hud(annotated, occupied=occupied, total=len(spots), width=width)

            if args.scale != 1.0:
                annotated = cv2.resize(annotated, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(annotated)
            written += 1
            if written % 25 == 0:
                target = args.max_frames or total
                print(f"wrote frame {written}/{target}")
    finally:
        capture.release()
        writer.release()

    print(f"\nwrote {written} annotated frames to {args.output}")
    print("green = predicted free, red = predicted occupied, amber = raw vehicle detection")


if __name__ == "__main__":
    main()
