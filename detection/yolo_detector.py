"""
yolo_detector.py

YOLO-based beacon detector for AI-VISTA.

The detector converts YOLO bounding-box detections into the common
detection format expected by the Kalman tracker.

The model is allowed to run at a smaller inference resolution to improve
real-time processing speed while the original camera frame remains
640x480.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YOLODetector:

    def __init__(
        self,
        model_path: str = "models/yolo/beacon_best.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 320,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"\nYOLO model not found:\n"
                f"  {self.model_path}\n\n"
                f"Train the beacon model first."
            )

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> dict:

        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        if not results:
            return self._not_found()

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return self._not_found()

        best_index = None
        best_confidence = -1.0

        for i, box in enumerate(
            result.boxes
        ):

            confidence = float(
                box.conf[0].item()
            )

            if confidence > best_confidence:

                best_confidence = confidence
                best_index = i

        if best_index is None:
            return self._not_found()

        box = result.boxes[best_index]

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        height, width = frame_bgr.shape[:2]

        x1 = max(
            0.0,
            min(float(x1), width - 1),
        )

        x2 = max(
            0.0,
            min(float(x2), width - 1),
        )

        y1 = max(
            0.0,
            min(float(y1), height - 1),
        )

        y2 = max(
            0.0,
            min(float(y2), height - 1),
        )

        box_width = max(
            0.0,
            x2 - x1,
        )

        box_height = max(
            0.0,
            y2 - y1,
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        return {
            "found": True,
            "centroid": (
                float(center_x),
                float(center_y),
            ),
            "area_px": int(
                box_width
                * box_height
            ),
            "confidence": float(
                best_confidence
            ),
            "bbox": (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ),
        }

    @staticmethod
    def _not_found():

        return {
            "found": False,
            "centroid": None,
            "area_px": 0,
            "confidence": 0.0,
            "bbox": None,
        }