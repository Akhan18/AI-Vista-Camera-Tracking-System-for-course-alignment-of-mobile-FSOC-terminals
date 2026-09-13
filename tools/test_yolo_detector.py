"""
test_yolo_detector.py

Quick standalone test for the AI-VISTA YOLO beacon detector.
"""

import sys
from pathlib import Path

import cv2


# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import YOLODetector


def main():

    image_path = (
        PROJECT_ROOT
        / "data"
        / "yolo_beacon"
        / "images"
        / "val"
        / "val_00000.png"
    )

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    detector = YOLODetector(
        model_path=str(
            PROJECT_ROOT
            / "models"
            / "yolo"
            / "beacon_best.pt"
        ),
        confidence_threshold=0.25,
    )

    detection = detector.detect(frame)

    print("\n--- YOLO DETECTION TEST ---")
    print(f"Found:        {detection['found']}")
    print(f"Centroid:     {detection['centroid']}")
    print(f"Area:         {detection['area_px']} px")
    print(f"Confidence:   {detection['confidence']:.4f}")
    print(f"Bounding box: {detection['bbox']}")

    if detection["found"]:
        print("\nYOLO DETECTOR TEST: PASS")
    else:
        print("\nYOLO DETECTOR TEST: FAIL")


if __name__ == "__main__":
    main()