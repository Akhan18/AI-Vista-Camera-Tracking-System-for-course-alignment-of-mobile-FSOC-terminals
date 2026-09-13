"""
renderer.py

Renders the simulated FSOC beacon into the virtual camera image.

The camera uses its configured FOV and optical-axis position to determine
where the beacon appears in the image.
"""

import numpy as np
import cv2


def render_frame(
    camera,
    beacon_world_xy: tuple[float, float],
    beacon_width: int,
    beacon_height: int,
    beacon_color_bgr: tuple[int, int, int],
    background_value: int = 10,
) -> tuple[
    np.ndarray,
    tuple[float, float] | None,
]:
    """
    Render the current camera view as a BGR image.

    Returns
    -------
    frame:
        Simulated camera image.

    beacon_frame_xy:
        Beacon centre in camera-frame pixels, or None if the beacon
        is outside the camera's field of view.
    """

    res_w = camera.res_width
    res_h = camera.res_height

    frame = np.full(
        (res_h, res_w, 3),
        background_value,
        dtype=np.uint8,
    )

    bx, by = beacon_world_xy

    # Convert world position to camera-frame position.
    frame_x, frame_y = camera.world_to_frame(bx, by)

    half_w = beacon_width / 2
    half_h = beacon_height / 2

    top_left = (
        int(frame_x - half_w),
        int(frame_y - half_h),
    )

    bottom_right = (
        int(frame_x + half_w),
        int(frame_y + half_h),
    )

    # Check whether any part of the beacon is inside the image.
    visible = (
        bottom_right[0] >= 0
        and top_left[0] <= res_w
        and bottom_right[1] >= 0
        and top_left[1] <= res_h
    )

    if visible:
        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            beacon_color_bgr,
            thickness=-1,
        )

        return frame, (frame_x, frame_y)

    return frame, None