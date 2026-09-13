"""
AI-VISTA External MP4 Evaluator

Purpose:
    Process an external MP4 through the AI-VISTA beacon
    detection and tracking pipeline.

Pipeline:

    External MP4
        ↓
    YOLO detector
        ↓
    Kalman tracker
        ↓
    Coarse pointing command
        ↓
    Metrics / CSV / annotated MP4

Important:
    True centroid error is calculated only when a matching
    <video_name>_groundtruth.csv file exists.

    Without ground truth, the program reports detection,
    tracking, pointing offset, acquisition, loss and FPS,
    but does NOT claim a centroid accuracy value.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ------------------------------------------------------------
# AI-VISTA modules
# ------------------------------------------------------------

from src.detection.yolo_detector import YOLODetector
from src.detection.classical_detector import ClassicalDetector
from src.tracking.kalman_tracker import KalmanTracker


# ------------------------------------------------------------
# Ground-truth loader
# ------------------------------------------------------------

def load_ground_truth(video_path):
    """
    Load ground truth if it exists.

    Expected filename:

        video_name_groundtruth.csv

    Expected columns:

        frame_index
        beacon_frame_x
        beacon_frame_y
    """

    ground_truth_path = video_path.with_name(
        video_path.stem
        + "_groundtruth.csv"
    )

    if not ground_truth_path.exists():

        return None, ground_truth_path

    ground_truth = {}

    with open(
        ground_truth_path,
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            frame_index = int(
                row["frame_index"]
            )

            x_value = row[
                "beacon_frame_x"
            ]

            y_value = row[
                "beacon_frame_y"
            ]

            if (
                x_value == ""
                or y_value == ""
            ):
                continue

            try:

                x = float(x_value)
                y = float(y_value)

                ground_truth[
                    frame_index
                ] = (x, y)

            except ValueError:

                continue

    return (
        ground_truth,
        ground_truth_path,
    )


# ------------------------------------------------------------
# Detector creation
# ------------------------------------------------------------

def create_detector(args):

    if args.detector == "yolo":

        return YOLODetector(
            model_path=args.yolo_model,
            confidence_threshold=(
                args.yolo_confidence
            ),
            image_size=320,
        )

    return ClassicalDetector(
        threshold=args.detector_threshold
    )


# ------------------------------------------------------------
# Main evaluator
# ------------------------------------------------------------

def evaluate_video(args):

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        print()
        print(
            f"ERROR: Video not found:"
            f" {video_path}"
        )
        return

    detector = create_detector(
        args
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print()
        print(
            f"ERROR: Could not open video:"
            f" {video_path}"
        )
        return

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:

        source_fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        total_frames / source_fps
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth, ground_truth_path = (
        load_ground_truth(
            video_path
        )
    )

    # --------------------------------------------------------
    # Kalman tracker
    # --------------------------------------------------------

    tracker = KalmanTracker(
        dt=1.0 / source_fps,
        max_missed_frames=10,
        acquisition_frames=5,
    )

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    output_video = video_path.with_name(
        video_path.stem
        + "_"
        + args.detector
        + "_evaluated.mp4"
    )

    log_path = video_path.with_name(
        video_path.stem
        + "_"
        + args.detector
        + "_evaluation.csv"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    writer = None

    if args.save_video:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            source_fps,
            (width, height),
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    log_file = open(
        log_path,
        "w",
        newline="",
    )

    csv_writer = csv.writer(
        log_file
    )

    csv_writer.writerow([
        "frame_index",
        "time_s",
        "detected",
        "confidence",
        "detected_x",
        "detected_y",
        "tracked_x",
        "tracked_y",
        "groundtruth_x",
        "groundtruth_y",
        "centroid_error_px",
        "pointing_offset_x_px",
        "pointing_offset_y_px",
        "pointing_offset_px",
        "command",
        "status",
    ])

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    frame_index = 0

    detected_frames = 0
    tracking_frames = 0
    predicted_frames = 0
    lost_frames = 0

    acquisition_time = None

    previous_status = "UNINITIALIZED"

    reacquisition_count = 0

    reacquisition_times = []

    centroid_errors = []

    pointing_offsets = []

    center_x = width / 2.0
    center_y = height / 2.0

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print(
        "======================================"
    )
    print(
        " AI-VISTA EXTERNAL MP4 EVALUATOR"
    )
    print(
        "======================================"
    )

    print(
        f"Video:             {video_path}"
    )

    print(
        f"Resolution:        "
        f"{width}x{height}"
    )

    print(
        f"Source FPS:        "
        f"{source_fps:.2f}"
    )

    print(
        f"Frames:            "
        f"{total_frames}"
    )

    print(
        f"Duration:          "
        f"{duration:.2f}s"
    )

    print(
        f"Detector:          "
        f"{args.detector.upper()}"
    )

    if ground_truth is not None:

        print(
            f"Ground truth:      "
            f"{ground_truth_path}"
        )

        print(
            f"Valid GT frames:   "
            f"{len(ground_truth)}"
        )

    else:

        print(
            "Ground truth:      "
            "NOT PROVIDED"
        )

    print()

    # --------------------------------------------------------
    # Processing loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        timestamp = (
            frame_index
            / source_fps
        )

        # ----------------------------------------------------
        # Detection
        # ----------------------------------------------------

        detection = detector.detect(
            frame
        )

        found = bool(
            detection.get(
                "found",
                False,
            )
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        detected_x = None
        detected_y = None

        if found:

            detected_frames += 1

            centroid = (
                detection.get(
                    "centroid"
                )
            )

            if centroid is not None:

                detected_x = float(
                    centroid[0]
                )

                detected_y = float(
                    centroid[1]
                )

        # ----------------------------------------------------
        # Kalman
        # ----------------------------------------------------

        tracker_result = (
            tracker.update(
                detection
            )
        )

        status = tracker_result[
            "status"
        ]

        position = tracker_result[
            "position"
        ]

        tracked_x = None
        tracked_y = None

        if position is not None:

            tracked_x = float(
                position[0]
            )

            tracked_y = float(
                position[1]
            )

        # ----------------------------------------------------
        # Acquisition
        # ----------------------------------------------------

        if (
            status == "TRACKING"
            and acquisition_time
            is None
        ):

            acquisition_time = (
                timestamp
            )

        # ----------------------------------------------------
        # Tracking state
        # ----------------------------------------------------

        if status == "TRACKING":

            tracking_frames += 1

            if (
                previous_status
                == "LOST"
            ):

                reacquisition_count += 1

                reacquisition_times.append(
                    timestamp
                )

        elif status == "PREDICTED":

            predicted_frames += 1

        elif status == "LOST":

            lost_frames += 1

        previous_status = status

        # ----------------------------------------------------
        # Pointing offset
        #
        # This is NOT centroid error.
        #
        # It represents the current distance of the tracked
        # beacon from the image centre.
        # ----------------------------------------------------

        pointing_offset_x = None
        pointing_offset_y = None
        pointing_offset = None

        command = "NO_TARGET"

        if (
            tracked_x is not None
            and tracked_y is not None
        ):

            pointing_offset_x = (
                tracked_x
                - center_x
            )

            pointing_offset_y = (
                tracked_y
                - center_y
            )

            pointing_offset = float(
                np.sqrt(
                    pointing_offset_x ** 2
                    + pointing_offset_y ** 2
                )
            )

            pointing_offsets.append(
                pointing_offset
            )

            threshold = 3.0

            if pointing_offset < threshold:

                command = "CENTER"

            elif abs(
                pointing_offset_x
            ) > abs(
                pointing_offset_y
            ):

                if pointing_offset_x > 0:

                    command = "RIGHT"

                else:

                    command = "LEFT"

            else:

                if pointing_offset_y > 0:

                    command = "DOWN"

                else:

                    command = "UP"

        # ----------------------------------------------------
        # Ground-truth centroid error
        # ----------------------------------------------------

        gt_x = None
        gt_y = None
        centroid_error = None

        if ground_truth is not None:

            if (
                frame_index
                in ground_truth
            ):

                gt_x, gt_y = (
                    ground_truth[
                        frame_index
                    ]
                )

                if (
                    tracked_x is not None
                    and tracked_y is not None
                ):

                    centroid_error = float(
                        np.sqrt(
                            (
                                tracked_x
                                - gt_x
                            ) ** 2
                            +
                            (
                                tracked_y
                                - gt_y
                            ) ** 2
                        )
                    )

                    centroid_errors.append(
                        centroid_error
                    )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_writer.writerow([
            frame_index,
            timestamp,
            int(found),
            confidence,
            detected_x,
            detected_y,
            tracked_x,
            tracked_y,
            gt_x,
            gt_y,
            centroid_error,
            pointing_offset_x,
            pointing_offset_y,
            pointing_offset,
            command,
            status,
        ])

        # ----------------------------------------------------
        # Annotated video
        # ----------------------------------------------------

        if writer is not None:

            annotated = frame.copy()

            # Image centre
            cv2.drawMarker(
                annotated,
                (
                    int(center_x),
                    int(center_y),
                ),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                18,
                1,
            )

            # YOLO bounding box
            bbox = detection.get(
                "bbox"
            )

            if bbox is not None:

                x1, y1, x2, y2 = [
                    int(v)
                    for v in bbox
                ]

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )

            # Detection centroid
            if (
                detected_x is not None
                and detected_y is not None
            ):

                cv2.circle(
                    annotated,
                    (
                        int(detected_x),
                        int(detected_y),
                    ),
                    4,
                    (255, 0, 0),
                    -1,
                )

            # Kalman position
            if (
                tracked_x is not None
                and tracked_y is not None
            ):

                cv2.circle(
                    annotated,
                    (
                        int(tracked_x),
                        int(tracked_y),
                    ),
                    7,
                    (0, 255, 255),
                    2,
                )

                cv2.line(
                    annotated,
                    (
                        int(center_x),
                        int(center_y),
                    ),
                    (
                        int(tracked_x),
                        int(tracked_y),
                    ),
                    (0, 255, 255),
                    1,
                )

            # Ground truth
            if (
                gt_x is not None
                and gt_y is not None
            ):

                cv2.drawMarker(
                    annotated,
                    (
                        int(gt_x),
                        int(gt_y),
                    ),
                    (255, 255, 255),
                    cv2.MARKER_TILTED_CROSS,
                    14,
                    1,
                )

            # Status
            cv2.putText(
                annotated,
                f"YOLO | {status}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Confidence
            cv2.putText(
                annotated,
                (
                    f"Confidence: "
                    f"{confidence:.2f}"
                ),
                (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Centroid error
            if centroid_error is not None:

                cv2.putText(
                    annotated,
                    (
                        f"Centroid error: "
                        f"{centroid_error:.2f}px"
                    ),
                    (8, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Pointing command
            cv2.putText(
                annotated,
                (
                    f"Pointing: "
                    f"{command}"
                ),
                (8, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Offset
            if pointing_offset is not None:

                cv2.putText(
                    annotated,
                    (
                        f"Offset: "
                        f"{pointing_offset:.1f}px"
                    ),
                    (8, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(
                annotated
            )

        frame_index += 1

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    end_time = time.perf_counter()

    cap.release()

    if writer is not None:

        writer.release()

    log_file.close()

    wall_time = (
        end_time
        - start_time
    )

    processing_fps = (
        frame_index
        / wall_time
        if wall_time > 0
        else 0.0
    )

    detection_rate = (
        detected_frames
        / frame_index
        * 100
        if frame_index > 0
        else 0.0
    )

    target_loss = (
        lost_frames
        / frame_index
        * 100
        if frame_index > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Centroid metrics
    # --------------------------------------------------------

    if centroid_errors:

        avg_centroid_error = float(
            np.mean(
                centroid_errors
            )
        )

        max_centroid_error = float(
            np.max(
                centroid_errors
            )
        )

        rmse_centroid_error = float(
            np.sqrt(
                np.mean(
                    np.square(
                        centroid_errors
                    )
                )
            )
        )

    else:

        avg_centroid_error = None
        max_centroid_error = None
        rmse_centroid_error = None

    # --------------------------------------------------------
    # Pointing metrics
    # --------------------------------------------------------

    if pointing_offsets:

        avg_pointing_offset = float(
            np.mean(
                pointing_offsets
            )
        )

        max_pointing_offset = float(
            np.max(
                pointing_offsets
            )
        )

    else:

        avg_pointing_offset = None
        max_pointing_offset = None

    # --------------------------------------------------------
    # Reacquisition
    # --------------------------------------------------------

    if reacquisition_times:

        avg_reacquisition = float(
            np.mean(
                reacquisition_times
            )
        )

    else:

        avg_reacquisition = None

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "======================================"
    )
    print(
        " MP4 EVALUATION SUMMARY"
    )
    print(
        "======================================"
    )

    print(
        f"Frames processed:      "
        f"{frame_index}"
    )

    print(
        f"Detection rate:        "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Processing FPS:        "
        f"{processing_fps:.2f}"
    )

    print(
        f"Acquisition time:      "
        f"{acquisition_time}"
    )

    print()

    print(
        "--- CENTROID METRICS ---"
    )

    if ground_truth is not None:

        print(
            f"GT frames:             "
            f"{len(ground_truth)}"
        )

        print(
            f"Compared frames:      "
            f"{len(centroid_errors)}"
        )

        print(
            f"Average error:         "
            f"{avg_centroid_error}"
        )

        print(
            f"Maximum error:         "
            f"{max_centroid_error}"
        )

        print(
            f"RMSE:                  "
            f"{rmse_centroid_error}"
        )

    else:

        print(
            "Ground truth:          NOT PROVIDED"
        )

        print(
            "Centroid error:        NOT CALCULATED"
        )

    print()

    print(
        "--- POINTING METRICS ---"
    )

    print(
        f"Average offset:        "
        f"{avg_pointing_offset}"
    )

    print(
        f"Maximum offset:        "
        f"{max_pointing_offset}"
    )

    print()

    print(
        "--- TRACKING METRICS ---"
    )

    print(
        f"Tracking frames:       "
        f"{tracking_frames}"
    )

    print(
        f"Predicted frames:      "
        f"{predicted_frames}"
    )

    print(
        f"Lost frames:           "
        f"{lost_frames}"
    )

    print(
        f"Target loss:           "
        f"{target_loss:.2f}%"
    )

    print(
        f"Reacquisitions:        "
        f"{reacquisition_count}"
    )

    print(
        f"Avg reacquisition:     "
        f"{avg_reacquisition}"
    )

    print()

    print(
        f"Evaluation CSV saved:  "
        f"{log_path}"
    )

    if writer is not None:

        print(
            f"Annotated MP4 saved:   "
            f"{output_video}"
        )

    print(
        "======================================"
    )


# ------------------------------------------------------------
# Command line
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI-VISTA external MP4 evaluator"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to evaluator MP4",
    )

    parser.add_argument(
        "--detector",
        choices=[
            "yolo",
            "classical",
        ],
        default="yolo",
        help="Beacon detector",
    )

    parser.add_argument(
        "--yolo-model",
        default=(
            "models/yolo/"
            "beacon_best.pt"
        ),
        help="YOLO model path",
    )

    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--detector-threshold",
        type=int,
        default=100,
        help=(
            "Threshold for classical "
            "detector"
        ),
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated MP4",
    )

    args = parser.parse_args()

    evaluate_video(
        args
    )


if __name__ == "__main__":

    main()