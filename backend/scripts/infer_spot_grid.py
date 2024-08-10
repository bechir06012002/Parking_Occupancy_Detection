"""Infer a full-lot spot layout from detections (demo visualization only).

`scripts/ground_truth/parking_lot.json` (20 spots) is the **accuracy**
ground truth — hand-verified, used to measure the 100% result in
docs/evaluation.md. It stays exactly as is; this script does not touch it
and does not attach an accuracy claim to what it produces.

What this script does: run the real `YoloDetector` (the same aerial OBB
checkpoint the pipeline uses, config-driven from `Settings`) on one frame,
then infer a full-lot spot layout for `scripts/visualize_occupancy.py` to
draw, so the demo video shows 200+ tightly-fitted stalls instead of the 20
in the accuracy set. This is a visualization convenience, not a second
production annotation path — the real one is `scripts/annotate_spots.py`,
still the only source of truth `parking_spots` rows are written from.

The grid is not a fixed overlay: every row/column's pitch and stall size is
derived from that row's own detected vehicles, not a global constant.

Algorithm
---------
1. Detect every vehicle on the reference frame.
2. Band detections into rows (near-constant y, varying x) and, for
   whatever's left over, columns (near-constant x, varying y) — this
   footage has both: horizontal rows in the main lot and a vertical
   perimeter column along the left/right building edges.
3. Split each band into spatially-continuous runs along its own axis — two
   cars can share a band's cross-axis position by coincidence (a car in a
   turning apron or access lane, not a real neighbor in that row) without
   being close together along the row itself; this stops a real row's box
   from stretching across empty ground to reach a stray member. Runs
   shorter than MIN_ROW_MEMBERS are dropped — a 2-3-member grouping is more
   likely noise than a real row.
4. Within each surviving run, derive a local pitch (median adjacent
   spacing) and a cross-axis band (median stall size) from its own members.
5. Walk consecutive members; where the gap is > ~1.5 pitches, insert
   evenly-spaced synthetic (empty) stall positions to fill it.
6. Emit one polygon per position: a real detection's own tight bbox where a
   car sits there, a row-statistics-derived box for a synthetic position.
7. By default, a detection that never joined a real run gets **no** spot
   box — it's almost always a car in transit through a lane or apron, not a
   parked one, which is exactly what this filtering is for. Pass
   `--include-singletons` to box it anyway (its own bbox, always occupied
   when the layout was built).

Usage:
    uv run python scripts/infer_spot_grid.py \
        --video ../data/parking_lot.mov \
        --output scripts/spot_layouts/parking_lot_full.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import get_settings
from app.domain.detection import Detection
from app.services.detection.yolo import YoloDetector

MIN_ROW_MEMBERS = 4  # a coincidental 2-3-member grouping is noise, not a real row of stalls
MAX_GAP_STEPS = 6  # cap on synthetic stalls inserted into a single gap
GAP_FACTOR = 1.5  # a consecutive spacing above pitch * this is "more than one stall wide"
SPLIT_GAP_FACTOR = 3.5  # along-axis gap above this multiple of car size ends a row/column run
# A run's *average* spacing (span / (members - 1)) relative to its own car
# size — every real stall row/column measured on this footage sits at
# 0.4-1.8x; a handful of cars stopped or parked loosely across an open
# drive lane (still individually under SPLIT_GAP_FACTOR so they survive
# continuity-splitting) averages 2.5x+. Reject the whole run above this,
# rather than drawing stall boxes across ground that was never striped.
MAX_DENSITY_RATIO = 2.0
STALL_MARGIN = 1.12  # small padding around a real detection's own bbox
SYNTHETIC_WIDTH_FRAC = 0.86  # narrower than full pitch so synthetic stalls show a visible gap

# Manually confirmed non-stall locations on `data/parking_lot.mov` frame 0 —
# an exit-lane arrow marking and the paved apron beside the diagonal-parked
# corner — that the clustering heuristics above cannot reliably tell apart
# from a real sparse row/column from geometry alone (see docs/evaluation.md
# "Full-lot demo layout" for why: a handful of cars sharing a row's height
# by coincidence looks the same as a few real cars with real empty stalls
# between them). Identified by a human reviewing the rendered output twice
# and pointing at the exact boxes; this is a targeted, video-specific
# correction, not a general rule — it has no effect on any other video.
# (cx, cy, radius) in frame-0 pixel coordinates.
MANUAL_EXCLUSION_ZONES: tuple[tuple[float, float, float], ...] = (
    (1854.0, 734.5, 20.0),
    (1811.0, 734.5, 20.0),
    (1780.0, 734.5, 16.0),
    (308.0, 978.5, 20.0),
    (1790.0, 1005.0, 20.0),
    (1867.0, 1005.0, 20.0),
)


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


def _to_dets(detections: list[Detection]) -> list[Det]:
    return [Det(*d.bbox) for d in detections]


def _segment(items: list[Det], *, key: Callable[[Det], float], tol: float) -> list[list[Det]]:
    """Split items (pre-sorted by `key`) wherever a consecutive gap exceeds `tol`."""
    segments: list[list[Det]] = []
    current: list[Det] = []
    for d in items:
        if current and key(d) - key(current[-1]) > tol:
            segments.append(current)
            current = []
        current.append(d)
    if current:
        segments.append(current)
    return segments


def _split_by_continuity(cluster: list[Det], *, axis: str) -> list[list[Det]]:
    """Break a cross-axis band into spatially-continuous runs along its own axis.

    Two cars can share a cross-axis band (near-identical y for a "row", near-
    identical x for a "column") by coincidence of camera geometry without
    belonging to the same physical row of stalls — a car in a turning apron
    or an access lane, sharing a row's height purely because of where the
    camera happens to put it. Splitting further wherever the along-axis gap
    is large relative to that band's own car size stops a real row's box
    from stretching across that empty ground to reach it; the stray member
    ends up in its own short run instead, which `MIN_ROW_MEMBERS` then
    drops entirely (see `_cluster`).
    """
    along_key = (lambda d: d.cx) if axis == "row" else (lambda d: d.cy)
    size_key = (lambda d: d.w) if axis == "row" else (lambda d: d.h)
    ordered = sorted(cluster, key=along_key)
    size_med = statistics.median(size_key(d) for d in ordered)
    return _segment(ordered, key=along_key, tol=SPLIT_GAP_FACTOR * size_med)


def _is_dense_run(run: list[Det], *, axis: str) -> bool:
    """Reject a run whose members are, on average, too spread out to be real stalls.

    A handful of cars stopped or parked loosely across an open drive lane
    can still pass `_split_by_continuity` (each individual gap under
    SPLIT_GAP_FACTOR) while being, on average, much sparser than any real
    packed row — see MAX_DENSITY_RATIO.
    """
    if len(run) < 2:
        return True
    along_key = (lambda d: d.cx) if axis == "row" else (lambda d: d.cy)
    size_key = (lambda d: d.w) if axis == "row" else (lambda d: d.h)
    alongs = sorted(along_key(d) for d in run)
    avg_pitch = (alongs[-1] - alongs[0]) / (len(run) - 1)
    size_med = statistics.median(size_key(d) for d in run)
    return size_med > 0 and avg_pitch / size_med <= MAX_DENSITY_RATIO


def _cluster(dets: list[Det], *, axis: str, tol: float) -> tuple[list[list[Det]], list[Det]]:
    """Group dets into real rows/columns of stalls, two-stage.

    axis="row": group by cy (horizontal rows, cars side by side varying in x).
    axis="col": group by cx (vertical columns, cars stacked varying in y).

    Stage 1 bands by cross-axis proximity (consecutive-neighbor-gap
    segmentation, not "distance to a cluster's running mean" — a mean-based
    greedy join lets a cluster's centroid drift step by step until it
    silently chains together two unrelated rows). Stage 2
    (`_split_by_continuity`) then breaks each band into spatially-continuous
    runs along its own axis, so a stray member sharing the band by
    coincidence doesn't stretch a real row's box out to reach it. Stage 3
    (`_is_dense_run`) drops whatever's left that's too sparse, on average,
    to be a real row of stalls.
    Returns (runs with >= MIN_ROW_MEMBERS members, leftover ungrouped dets).
    """
    key = (lambda d: d.cy) if axis == "row" else (lambda d: d.cx)
    bands = _segment(sorted(dets, key=key), key=key, tol=tol)
    runs = [run for band in bands for run in _split_by_continuity(band, axis=axis)]
    kept = [r for r in runs if len(r) >= MIN_ROW_MEMBERS and _is_dense_run(r, axis=axis)]
    leftover = [d for d in dets if not any(d in c for c in kept)]
    return kept, leftover


def _pitch(alongs: list[float], sizes: list[float]) -> float:
    gaps = [b - a for a, b in zip(alongs, alongs[1:], strict=False)]
    size_med = statistics.median(sizes)
    adjacent_gaps = [g for g in gaps if g <= GAP_FACTOR * size_med * 1.4] or gaps
    return statistics.median(adjacent_gaps) if adjacent_gaps else size_med * 1.05


def _line_polygon(
    positions: list[tuple[float, Det | None]],
    *,
    axis: str,
    band_lo: float,
    band_hi: float,
    synthetic_size: float,
) -> list[list[list[float]]]:
    """Build one axis-aligned polygon per grid position.

    axis="row": along-axis is x, band is the row's y-extent.
    axis="col": along-axis is y, band is the column's x-extent.
    """
    polys: list[list[list[float]]] = []
    for along, det in positions:
        if det is not None:
            # Real detection: use its own tight bbox on the along-axis, padded a
            # touch; band comes from the row/column so every stall in it lines up.
            if axis == "row":
                half = (det.x2 - det.x1) / 2 * STALL_MARGIN
                lo, hi = det.cx - half, det.cx + half
            else:
                half = (det.y2 - det.y1) / 2 * STALL_MARGIN
                lo, hi = det.cy - half, det.cy + half
        else:
            half = synthetic_size / 2 * SYNTHETIC_WIDTH_FRAC
            lo, hi = along - half, along + half

        if axis == "row":
            x1, x2, y1, y2 = lo, hi, band_lo, band_hi
        else:
            x1, x2, y1, y2 = band_lo, band_hi, lo, hi
        polys.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return polys


def _build_grid(members: list[Det], *, axis: str) -> list[list[list[float]]]:
    key = (lambda d: d.cx) if axis == "row" else (lambda d: d.cy)
    size_key = (lambda d: d.w) if axis == "row" else (lambda d: d.h)
    band_lo_key = (lambda d: d.y1) if axis == "row" else (lambda d: d.x1)
    band_hi_key = (lambda d: d.y2) if axis == "row" else (lambda d: d.x2)

    ordered = sorted(members, key=key)
    alongs = [key(d) for d in ordered]
    sizes = [size_key(d) for d in ordered]
    pitch = _pitch(alongs, sizes)

    band_center = statistics.median((band_lo_key(d) + band_hi_key(d)) / 2 for d in ordered)
    band_size = statistics.median(band_hi_key(d) - band_lo_key(d) for d in ordered) * STALL_MARGIN
    band_lo, band_hi = band_center - band_size / 2, band_center + band_size / 2

    positions: list[tuple[float, Det | None]] = [(alongs[0], ordered[0])]
    for prev_along, det, along in zip(alongs, ordered[1:], alongs[1:], strict=False):
        gap = along - prev_along
        steps = max(1, round(gap / pitch)) if pitch > 0 else 1
        # A huge projected step count means the pitch estimate (or the row
        # itself) is unreliable for this particular gap — don't manufacture
        # a long run of synthetic stalls from a bad estimate, just skip the
        # gap (leave it unfilled) rather than flooding the layout.
        if steps > MAX_GAP_STEPS:
            positions.append((along, det))
            continue
        step = gap / steps
        for k in range(1, steps + 1):
            pos = prev_along + step * k
            is_real_endpoint = k == steps
            positions.append((pos, det if is_real_endpoint else None))

    return _line_polygon(
        positions,
        axis=axis,
        band_lo=band_lo,
        band_hi=band_hi,
        synthetic_size=statistics.median(sizes),
    )


def _drop_manual_exclusions(
    polygons: list[list[list[float]]],
) -> list[list[list[float]]]:
    kept = []
    for poly in polygons:
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        if any(
            (cx - zx) ** 2 + (cy - zy) ** 2 <= radius**2
            for zx, zy, radius in MANUAL_EXCLUSION_ZONES
        ):
            continue
        kept.append(poly)
    return kept


def infer_layout(dets: list[Det], *, include_singletons: bool = False) -> list[list[list[float]]]:
    global_h = statistics.median(d.h for d in dets)
    global_w = statistics.median(d.w for d in dets)

    # Consecutive-neighbor-gap tolerance: wide enough to absorb the normal
    # spread of cy (resp. cx) within one real row/column, narrow enough not
    # to bridge the gap into the next one. ~0.28x a car's own size is where
    # that boundary actually falls on this footage (swept empirically —
    # 0.55x silently merges adjacent rows into one via chained small gaps).
    row_clusters, leftover = _cluster(dets, axis="row", tol=0.28 * global_h)
    col_clusters, leftover = _cluster(leftover, axis="col", tol=0.28 * global_w)

    polygons: list[list[list[float]]] = []
    for row in row_clusters:
        polygons.extend(_build_grid(row, axis="row"))
    for col in col_clusters:
        polygons.extend(_build_grid(col, axis="col"))

    # A detection that never joined a real row/column run (of >= a few
    # members) is almost always a car passing through an access lane,
    # turning apron, or entrance ramp rather than a parked one — the exact
    # complaint this filtering exists for. Off by default: no spot box for
    # it, only its (unaffected) detection box. Pass --include-singletons to
    # get the old best-effort behavior back.
    if include_singletons:
        for d in leftover:
            half_w, half_h = (d.x2 - d.x1) / 2 * STALL_MARGIN, (d.y2 - d.y1) / 2 * STALL_MARGIN
            polygons.append(
                [
                    [d.cx - half_w, d.cy - half_h],
                    [d.cx + half_w, d.cy - half_h],
                    [d.cx + half_w, d.cy + half_h],
                    [d.cx - half_w, d.cy + half_h],
                ]
            )
    return _drop_manual_exclusions(polygons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include-singletons",
        action="store_true",
        help="Also box detections that never joined a real row/column (usually cars in "
        "access lanes/aprons, not parked ones) — off by default",
    )
    args = parser.parse_args()

    settings = get_settings()
    detector = YoloDetector(
        weights_path=settings.yolo_weights_path,
        confidence_threshold=settings.yolo_conf_threshold,
        imgsz=settings.yolo_imgsz,
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    frame = None
    for _ in range(args.frame_index + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {args.frame_index}")
    capture.release()
    assert frame is not None

    [detections] = detector.predict_batch([frame])
    dets = _to_dets(detections)
    print(f"detections on frame {args.frame_index}: {len(dets)}")

    polygons = infer_layout(dets, include_singletons=args.include_singletons)
    spots = [{"label": f"S{i + 1}", "polygon": poly} for i, poly in enumerate(polygons)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "video": str(args.video),
                "frame_index": args.frame_index,
                "note": (
                    "Auto-inferred full-lot spot layout for scripts/visualize_occupancy.py "
                    "ONLY — not the accuracy ground truth (see "
                    "scripts/ground_truth/parking_lot.json for that) and carries no accuracy "
                    "claim of its own. Generated by scripts/infer_spot_grid.py: row/column "
                    "clustering of real detections, per-row/column pitch and stall size, gaps "
                    "filled with synthetic (empty) stall positions."
                ),
                "spots": spots,
            },
            indent=2,
        )
    )
    print(f"wrote {len(spots)} spots to {args.output}")


if __name__ == "__main__":
    main()
