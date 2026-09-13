"""
generate_clean_video.py

Generates a clean AI-VISTA simulation MP4 for evaluation.

The video contains:
- only the simulated camera image
- no YOLO boxes
- no tracking overlay
- no controller text

A CSV file containing the known beacon centroid for every frame
is also generated.
"""

import sys
import csv
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_scenario
from src.simulation.target import Beacon
from src.simulation.renderer import render_frame
from src.camera.virtual_camera import VirtualCamera


def main():

    scenario = "circular"

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "default.yaml"
    )

    scenario_path = (
        PROJECT_ROOT
        / "configs"
        / f"{scenario}.yaml"
    )

    cfg = load_scenario(
        config_path,
        scenario_path,
    )

    duration_s = 15.0

    fps = cfg.camera.update_rate_hz

    dt = 1.0 / fps

    frame_count = int(
        duration_s * fps
    )

    beacon = Beacon(cfg)

    camera = VirtualCamera.from_config(
        cfg
    )

    # For the clean benchmark we keep the
    # virtual camera fixed at the centre of
    # the circular trajectory.
    camera.center_x = 1000
    camera.center_y = 1000

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = (
        output_dir
        / "clean_circular.mp4"
    )

    groundtruth_path = (
        output_dir
        / "clean_circular_groundtruth.csv"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        fps,
        (
            camera.res_width,
            camera.res_height,
        ),
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    valid_groundtruth_frames = 0

    with open(
        groundtruth_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        csv_writer = csv.writer(
            csv_file
        )

        csv_writer.writerow(
            [
                "frame_index",
                "time_s",
                "beacon_world_x",
                "beacon_world_y",
                "beacon_frame_x",
                "beacon_frame_y",
            ]
        )

        for frame_index in range(
            frame_count
        ):

            t = (
                frame_index * dt
            )

            beacon_world_xy = (
                beacon.position_at(
                    t,
                    dt,
                )
            )

            frame, frame_xy = (
                render_frame(
                    camera,
                    beacon_world_xy,
                    beacon.width,
                    beacon.height,
                    beacon.color_bgr,
                )
            )

            writer.write(
                frame
            )

            if frame_xy is not None:

                frame_x, frame_y = (
                    frame_xy
                )

                valid_groundtruth_frames += 1

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{t:.6f}",
                        f"{beacon_world_xy[0]:.6f}",
                        f"{beacon_world_xy[1]:.6f}",
                        f"{frame_x:.6f}",
                        f"{frame_y:.6f}",
                    ]
                )

            else:

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{t:.6f}",
                        f"{beacon_world_xy[0]:.6f}",
                        f"{beacon_world_xy[1]:.6f}",
                        "",
                        "",
                    ]
                )

    writer.release()

    print()
    print(
        "CLEAN EVALUATION VIDEO CREATED"
    )

    print(
        f"Video:       {video_path}"
    )

    print(
        f"Groundtruth: {groundtruth_path}"
    )

    print(
        f"FPS:         {fps}"
    )

    print(
        f"Frames:      {frame_count}"
    )

    print(
        f"Valid GT:    "
        f"{valid_groundtruth_frames}"
    )

    print(
        f"Duration:    {duration_s}s"
    )


if __name__ == "__main__":
    main()