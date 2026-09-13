"""
evaluate_yolo_groundtruth.py

Evaluates YOLO beacon centroiding against known ground-truth
centroids from a clean AI-VISTA simulation video.

Metrics:
- detection rate
- average centroid error
- maximum centroid error
- RMSE
- processing FPS
"""

import sys
import csv
import time
from pathlib import Path

import cv2
import numpy as np


# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.detection.yolo_detector import YOLODetector


def load_groundtruth(csv_path):

    groundtruth = {}

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            frame_index = int(
                row["frame_index"]
            )

            x_text = row[
                "beacon_frame_x"
            ]

            y_text = row[
                "beacon_frame_y"
            ]

            if (
                x_text == ""
                or y_text == ""
            ):

                groundtruth[
                    frame_index
                ] = None

            else:

                groundtruth[
                    frame_index
                ] = (
                    float(x_text),
                    float(y_text),
                )

    return groundtruth


def main():

    video_path = (
        PROJECT_ROOT
        / "data"
        / "output"
        / "clean_circular.mp4"
    )

    groundtruth_path = (
        PROJECT_ROOT
        / "data"
        / "output"
        / "clean_circular_groundtruth.csv"
    )

    model_path = (
        PROJECT_ROOT
        / "models"
        / "yolo"
        / "beacon_best.pt"
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not groundtruth_path.exists():

        raise FileNotFoundError(
            "Ground-truth CSV not found: "
            f"{groundtruth_path}"
        )

    if not model_path.exists():

        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    groundtruth = load_groundtruth(
        groundtruth_path
    )

    detector = YOLODetector(
        model_path=str(model_path),
        confidence_threshold=0.25,
        image_size=320,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    errors = []

    detected_frames = 0
    valid_frames = 0

    frame_index = 0

    start_time = time.perf_counter()

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        truth = groundtruth.get(
            frame_index
        )

        if truth is not None:

            valid_frames += 1

            detection = detector.detect(
                frame
            )

            if detection["found"]:

                detected_frames += 1

                predicted_x, predicted_y = (
                    detection["centroid"]
                )

                truth_x, truth_y = truth

                error = (
                    (
                        predicted_x
                        - truth_x
                    ) ** 2
                    +
                    (
                        predicted_y
                        - truth_y
                    ) ** 2
                ) ** 0.5

                errors.append(
                    error
                )

        frame_index += 1

    elapsed = (
        time.perf_counter()
        - start_time
    )

    cap.release()

    processing_fps = (
        frame_index
        / max(elapsed, 1e-9)
    )

    detection_rate = (
        detected_frames
        / max(valid_frames, 1)
        * 100.0
    )

    if errors:

        errors_array = np.asarray(
            errors,
            dtype=np.float64,
        )

        average_error = float(
            errors_array.mean()
        )

        maximum_error = float(
            errors_array.max()
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    errors_array ** 2
                )
            )
        )

    else:

        average_error = None
        maximum_error = None
        rmse = None

    print()
    print(
        "======================================"
    )
    print(
        " YOLO GROUND-TRUTH BENCHMARK"
    )
    print(
        "======================================"
    )

    print(
        f"Video:             {video_path.name}"
    )

    print(
        f"Source FPS:        {source_fps:.2f}"
    )

    print(
        f"Frames:            {frame_index}"
    )

    print(
        f"Valid GT frames:   {valid_frames}"
    )

    print(
        f"Detected frames:   {detected_frames}"
    )

    print(
        f"Detection rate:    {detection_rate:.2f}%"
    )

    print(
        f"Processing FPS:    {processing_fps:.2f}"
    )

    print(
        f"Average error:     {average_error}"
    )

    print(
        f"Maximum error:     {maximum_error}"
    )

    print(
        f"RMSE:              {rmse}"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()