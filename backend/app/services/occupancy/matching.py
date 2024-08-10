"""Occupancy matching (Strategy pattern).

Two interchangeable `OccupancyStrategy` implementations, both requiring a
track's bbox centroid to fall inside the spot polygon (centroid alone
mishandles angled/adjacent spots) plus a geometric-overlap gate:

- `CentroidCoverageStrategy` (the project default): the gate is
  "what fraction of the vehicle box lies inside the spot" ≥ threshold. A
  parking stall is drawn longer than a car — it includes manoeuvring
  clearance — so symmetric IoU(car, stall) tops out well below 1 even for a
  perfectly-parked car; coverage does not have that ceiling.
- `CentroidIoUStrategy` (the original, kept swappable): the gate is
  IoU(track bbox, spot polygon bbox) ≥ threshold.

Thresholds are required constructor parameters, never literals in this
module — the caller reads them from `Settings`. Pure functions/dataclasses
only — no I/O, no cv2/torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from app.domain.occupancy import Spot, SpotState
from app.domain.tracking import Track

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


class OccupancyStrategy(Protocol):
    def match(self, tracks: Sequence[Track], spots: Sequence[Spot]) -> list[SpotState]: ...


def _centroid(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def _polygon_bbox(polygon: Sequence[Point]) -> BBox:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: BBox, b: BBox) -> float:
    intersection = _intersection_area(a, b)
    union = _area(a) + _area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _coverage(vehicle: BBox, spot_bbox: BBox) -> float:
    """Fraction of the vehicle box that lies inside the spot's bounding box."""
    vehicle_area = _area(vehicle)
    return _intersection_area(vehicle, spot_bbox) / vehicle_area if vehicle_area > 0 else 0.0


def _match(
    tracks: Sequence[Track],
    spots: Sequence[Spot],
    *,
    score: Callable[[BBox, BBox], float],
    threshold: float,
) -> list[SpotState]:
    results: list[SpotState] = []
    for spot in spots:
        spot_bbox = _polygon_bbox(spot.polygon)
        best_track: Track | None = None
        best_score = 0.0
        for track in tracks:
            if not _point_in_polygon(_centroid(track.bbox), spot.polygon):
                continue
            value = score(track.bbox, spot_bbox)
            if value >= threshold and value > best_score:
                best_track = track
                best_score = value
        if best_track is not None:
            results.append(
                SpotState(
                    spot_id=spot.spot_id,
                    is_occupied=True,
                    confidence=best_track.confidence,
                    track_id=best_track.track_id,
                )
            )
        else:
            results.append(
                SpotState(spot_id=spot.spot_id, is_occupied=False, confidence=0.0, track_id=None)
            )
    return results


class CentroidCoverageStrategy:
    """Default OccupancyStrategy: centroid-in-polygon AND vehicle-coverage >= threshold."""

    def __init__(self, *, coverage_threshold: float) -> None:
        self._coverage_threshold = coverage_threshold

    def match(self, tracks: Sequence[Track], spots: Sequence[Spot]) -> list[SpotState]:
        return _match(tracks, spots, score=_coverage, threshold=self._coverage_threshold)


class CentroidIoUStrategy:
    """OccupancyStrategy: centroid-in-polygon AND bbox IoU >= threshold."""

    def __init__(self, *, iou_threshold: float) -> None:
        self._iou_threshold = iou_threshold

    def match(self, tracks: Sequence[Track], spots: Sequence[Spot]) -> list[SpotState]:
        return _match(tracks, spots, score=_iou, threshold=self._iou_threshold)
