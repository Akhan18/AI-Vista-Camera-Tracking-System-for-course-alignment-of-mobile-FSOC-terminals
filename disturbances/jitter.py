"""
jitter.py

Simulates unwanted camera/platform vibration -- random small displacement
of the camera's viewport each frame, ON TOP OF the intentional control-loop
movement from controller.py. This models a mobile FSOC terminal's platform
(e.g. vehicle, ship, drone) shaking the optics slightly.

Kept separate from controller.py deliberately: jitter is a DISTURBANCE we
want to measure robustness against, not a control action.
"""

import random


def apply_jitter(camera, max_pixels_per_frame: float = 20):
    """
    Nudge the camera's centre by a random offset, up to
    `max_pixels_per_frame` pixels in each axis. Call this once per frame,
    after the control step, before rendering.
    """
    dx = random.uniform(-max_pixels_per_frame, max_pixels_per_frame)
    dy = random.uniform(-max_pixels_per_frame, max_pixels_per_frame)

    camera.center_x = min(max(camera.center_x + dx, 0), camera.world_width)
    camera.center_y = min(max(camera.center_y + dy, 0), camera.world_height)
