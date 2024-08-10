from dataclasses import dataclass

# Vehicle class names kept after detection. Covers both label vocabularies
# the project's supported checkpoints emit:
#   - COCO (stock yolov8n.pt): car / truck / bus / motorcycle
#   - DOTA (yolov8n-obb.pt, the aerial default): "small vehicle" /
#     "large vehicle", with hyphenated variants seen in some DOTA configs.
# The vehicle-class filter lives in services/detection/yolo.py; this frozen
# set is the single source of what counts as a vehicle.
VEHICLE_CLASSES = frozenset(
    {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "small vehicle",
        "large vehicle",
        "small-vehicle",
        "large-vehicle",
    }
)


@dataclass(frozen=True)
class Detection:
    """One vehicle detection in a single frame, before tracking."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_name: str
