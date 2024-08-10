"""The one canonical frame type shared by capture/detection/pipeline.

A single definition avoids two structurally-different `Frame` aliases
silently failing Protocol conformance between adapters (mypy strict caught
exactly this once frame_grabber.py and yolo.py each had their own).
"""

from cv2.typing import MatLike

Frame = MatLike
