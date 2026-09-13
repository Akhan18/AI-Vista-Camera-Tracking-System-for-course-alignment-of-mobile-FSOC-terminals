"""
AI-VISTA application entry point.

Supports two detector modes:

    --detector yolo
    --detector classical

The YOLO detector is the AI-based detector.
The classical detector is retained as a fast baseline/fallback.

Pipeline:

    beacon motion
        ↓
    virtual camera
        ↓
    rendered frame
        ↓
    disturbances
        ↓
    YOLO / Classical detector
        ↓
    Kalman tracker
        ↓
    camera controller
        ↓
    performance metrics
        ↓
    logs / video
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.utils.config_loader import load_scenario
from src.simulation.environment import World
from src.simulation.target import Beacon
from src.simulation.renderer import render_frame
from src.camera.virtual_camera import VirtualCamera
from src.camera.controller import step_control
from src.detection.classical_detector import ClassicalDetector
from src.detection.yolo_detector import YOLODetector
from src.tracking.kalman_tracker import KalmanTracker
from src.disturbances import noise, atmosphere, jitter
from src.evaluation.metrics import MetricsCollector
from src.evaluation.logger import save_run_logs


def apply_disturbances(frame, cfg):
    """Apply every enabled disturbance from the configuration."""

    d = cfg.disturbances

    if d.gaussian_noise.enabled:
        frame = noise.add_gaussian_noise(
            frame,
            d.gaussian_noise.mean,
            d.gaussian_noise.stddev,
        )

    if d.salt_and_pepper_noise.enabled:
        frame = noise.add_salt_and_pepper_noise(
            frame,
            d.salt_and_pepper_noise.amount,
        )

    if d.poisson_noise.enabled:
        frame = noise.add_poisson_noise(frame)

    if d.low_light.enabled:
        frame = atmosphere.apply_low_light(
            frame,
            d.low_light.brightness_factor,
        )

    if d.haze.enabled:
        frame = atmosphere.apply_haze(
            frame,
            d.haze.intensity,
        )

    if d.fog.enabled:
        frame = atmosphere.apply_fog(
            frame,
            d.fog.intensity,
        )

    return frame


def draw_overlay(
    frame,
    camera,
    detection,
    tracker_result,
    control_result,
    detector_name,
):
    """Draw tracking and detector information on the frame."""

    annotated = frame.copy()

    res_w = camera.res_width
    res_h = camera.res_height

    cx = int(res_w / 2)
    cy = int(res_h / 2)

    # Image centre / optical axis
    cv2.drawMarker(
        annotated,
        (cx, cy),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        15,
        1,
    )

    # YOLO bounding box
    if detection.get("bbox") is not None:

        x1, y1, x2, y2 = detection["bbox"]

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1,
        )

    # Detection centroid
    if detection.get("centroid") is not None:

        dx = int(detection["centroid"][0])
        dy = int(detection["centroid"][1])

        cv2.circle(
            annotated,
            (dx, dy),
            5,
            (255, 0, 0),
            -1,
        )

    # Kalman tracked position
    if tracker_result["position"] is not None:

        tx = int(tracker_result["position"][0])
        ty = int(tracker_result["position"][1])

        if tracker_result["status"] == "TRACKING":
            tracking_color = (0, 255, 255)
        else:
            tracking_color = (0, 0, 255)

        cv2.circle(
            annotated,
            (tx, ty),
            8,
            tracking_color,
            1,
        )

        cv2.line(
            annotated,
            (cx, cy),
            (tx, ty),
            tracking_color,
            1,
        )

    # Status information
    status_text = (
        f"{detector_name} | "
        f"{tracker_result['status']}"
    )

    if control_result is not None:

        status_text += (
            f" | err="
            f"{control_result.error_magnitude_px:.1f}px"
            f" | {control_result.command}"
        )

    cv2.putText(
        annotated,
        status_text,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Detection confidence
    confidence_text = (
        f"confidence="
        f"{detection.get('confidence', 0.0):.3f}"
    )

    cv2.putText(
        annotated,
        confidence_text,
        (5, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


def create_detector(
    detector_name,
    detector_threshold,
    yolo_model,
    yolo_confidence,
):
    """Create the requested detector."""

    if detector_name == "yolo":

        print("Detector: YOLO")
        print(f"YOLO model: {yolo_model}")
        print(
            f"YOLO confidence threshold: "
            f"{yolo_confidence}"
        )

        return YOLODetector(
            model_path=yolo_model,
            confidence_threshold=yolo_confidence,
        )

    if detector_name == "classical":

        print("Detector: Classical CV")

        return ClassicalDetector(
            threshold=detector_threshold,
            use_otsu=True,
        )

    raise ValueError(
        f"Unknown detector '{detector_name}'. "
        f"Use 'yolo' or 'classical'."
    )


def run(
    scenario: str,
    base_config: str,
    duration_override: float | None,
    save_video: bool,
    gain: float,
    detector_name: str,
    detector_threshold: int,
    yolo_model: str,
    yolo_confidence: float,
):

    base_path = Path(base_config)
    scenario_path = Path("configs") / f"{scenario}.yaml"

    cfg = load_scenario(
        base_path,
        scenario_path,
    )

    duration_s = (
        duration_override
        if duration_override is not None
        else cfg.simulation.duration_s
    )

    dt = 1.0 / cfg.camera.update_rate_hz

    n_frames = int(
        duration_s
        * cfg.camera.update_rate_hz
    )

    world = World.from_config(cfg)

    beacon = Beacon(cfg)

    camera = VirtualCamera.from_config(cfg)

    detector = create_detector(
        detector_name=detector_name,
        detector_threshold=detector_threshold,
        yolo_model=yolo_model,
        yolo_confidence=yolo_confidence,
    )

    tracker = KalmanTracker(
        dt=dt
    )

    metrics = MetricsCollector(
        performance_targets=(
            cfg.performance_targets.to_dict()
        )
    )

    video_writer = None

    if save_video:

        out_dir = Path(
            cfg.paths.output_dir
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        video_path = (
            out_dir
            / f"{scenario}_{detector_name}_annotated.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            cfg.camera.update_rate_hz,
            (
                cfg.camera.resolution.width_px,
                cfg.camera.resolution.height_px,
            ),
        )

    print()

    print(
        f"Running scenario '{scenario}' "
        f"({cfg.meta.description})"
    )

    print(
        f"  duration={duration_s}s, "
        f"dt={dt:.4f}s, "
        f"frames={n_frames}"
    )

    print(
        f"  motion={cfg.motion.type}"
    )

    print()

    for frame_index in range(n_frames):

        t = frame_index * dt

        frame_start = time.perf_counter()

        beacon_world_xy = (
            beacon.position_at(
                t,
                dt,
            )
        )

        # Camera jitter
        if (
            cfg.disturbances
            .camera_jitter
            .enabled
        ):

            jitter.apply_jitter(
                camera,
                cfg.disturbances
                .camera_jitter
                .max_pixels_per_frame,
            )

        # Target occlusion
        occlusion = getattr(
            cfg.disturbances,
            "target_occlusion",
            None,
        )

        target_hidden = (
            occlusion is not None
            and occlusion.enabled
            and occlusion.start_s <= t
            < (
                occlusion.start_s
                + occlusion.duration_s
            )
        )

        if target_hidden:

            frame = np.full(
                (
                    camera.res_height,
                    camera.res_width,
                    3,
                ),
                10,
                dtype=np.uint8,
            )

            groundtruth = None

        else:

            frame, groundtruth = (
                render_frame(
                    camera,
                    beacon_world_xy,
                    beacon.width,
                    beacon.height,
                    beacon.color_bgr,
                )
            )

        # Apply disturbances
        frame = apply_disturbances(
            frame,
            cfg,
        )

        # Detection
        detection = detector.detect(
            frame
        )

        # Tracking
        tracker_result = tracker.update(
            detection
        )

        # Camera control
        control_result = None

        if (
            tracker_result["position"]
            is not None
        ):

            control_result = step_control(
                camera,
                tracker_result["position"],
                dt,
                gain=gain,
            )

        processing_time_s = (
            time.perf_counter()
            - frame_start
        )

        # Metrics
        metrics.record_frame(
            frame_index=frame_index,
            t=t,
            control_result=control_result,
            tracker_status=(
                tracker_result["status"]
            ),
            detection_found=(
                detection["found"]
            ),
            processing_time_s=(
                processing_time_s
            ),
        )

        # Save annotated video
        if video_writer is not None:

            annotated = draw_overlay(
                frame,
                camera,
                detection,
                tracker_result,
                control_result,
                detector_name,
            )

            video_writer.write(
                annotated
            )

    if video_writer is not None:

        video_writer.release()

    summary = metrics.summary()

    run_dir = save_run_logs(
        metrics,
        cfg.paths.output_dir,
        scenario,
    )

    print()

    print(
        "--- PERFORMANCE SUMMARY "
        "(measured, not fabricated) ---"
    )

    for key, value in summary.items():

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        f"Logs saved to: {run_dir}"
    )

    if save_video:

        video_filename = (
            f"{scenario}_{detector_name}"
            "_annotated.mp4"
        )

        print(
            "Annotated video saved to: "
            f"{Path(cfg.paths.output_dir) / video_filename}"
        )

    return summary, run_dir


def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI-VISTA FSOC simulation runner"
        )
    )

    parser.add_argument(
        "--scenario",
        default="default",
        help=(
            "Scenario name, matching "
            "configs/<scenario>.yaml"
        ),
    )

    parser.add_argument(
        "--base-config",
        default="configs/default.yaml",
        help="Base configuration file",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=(
            "Override simulation duration "
            "in seconds"
        ),
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help=(
            "Save annotated simulation video"
        ),
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=0.5,
        help=(
            "Proportional camera-control gain"
        ),
    )

    parser.add_argument(
        "--detector",
        choices=[
            "yolo",
            "classical",
        ],
        default="yolo",
        help=(
            "Beacon detector to use"
        ),
    )

    parser.add_argument(
        "--detector-threshold",
        type=int,
        default=100,
        help=(
            "Brightness threshold for "
            "classical detector"
        ),
    )

    parser.add_argument(
        "--yolo-model",
        default=(
            "models/yolo/"
            "beacon_best.pt"
        ),
        help="Path to YOLO model",
    )

    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help=(
            "YOLO confidence threshold"
        ),
    )

    args = parser.parse_args()

    run(
        scenario=args.scenario,
        base_config=args.base_config,
        duration_override=args.duration,
        save_video=args.save_video,
        gain=args.gain,
        detector_name=args.detector,
        detector_threshold=(
            args.detector_threshold
        ),
        yolo_model=args.yolo_model,
        yolo_confidence=(
            args.yolo_confidence
        ),
    )


if __name__ == "__main__":
    main()