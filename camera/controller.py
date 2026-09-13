"""
controller.py

Implements the core control loop step described in the problem statement:

    error_x = target_x - camera_center_x
    error_y = target_y - camera_center_y

...then converts that error into a movement command, and moves the virtual
camera to reduce the error over time (coarse alignment).

CONTROL LAW: simple proportional (P) controller. error is in camera-frame
pixel coordinates (i.e. relative to the image centre); we move the camera
centre in WORLD coordinates by a fraction (gain) of that error each frame.
A pure P-controller is intentionally the starting point -- it is easy to
reason about and tune. If overshoot/oscillation shows up in testing, a
PID or rate-limited variant can be justified with evidence, not assumed
upfront.
"""

from dataclasses import dataclass


@dataclass
class ControlResult:
    error_x: float
    error_y: float
    error_magnitude_px: float
    command: str  # "LEFT" | "RIGHT" | "UP" | "DOWN" | "CENTER"


# Below this pixel error, we consider the beacon "centred" / locked.
LOCK_THRESHOLD_PX = 3.0


def compute_error(target_frame_xy: tuple[float, float],
                   camera_image_center: tuple[float, float]) -> tuple[float, float]:
    """error = target position - image centre, both in camera-frame pixels."""
    tx, ty = target_frame_xy
    cx, cy = camera_image_center
    return tx - cx, ty - cy


def error_to_command(error_x: float, error_y: float) -> str:
    """
    Turn a 2D pixel error into a simple discrete direction command.
    This is mainly for human-readable logging/demo purposes -- the actual
    camera movement uses the continuous (error_x, error_y), not this label.
    """
    magnitude = (error_x ** 2 + error_y ** 2) ** 0.5
    if magnitude < LOCK_THRESHOLD_PX:
        return "CENTER"

    # Dominant axis decides the label (simple, readable for a demo).
    if abs(error_x) > abs(error_y):
        return "RIGHT" if error_x > 0 else "LEFT"
    return "DOWN" if error_y > 0 else "UP"


def step_control(camera, target_frame_xy: tuple[float, float], dt: float,
                  gain: float = 0.5) -> ControlResult:
    """
    One full control-loop step: compute error, derive a command, and move
    the camera toward the target (proportional control).

    Parameters
    ----------
    camera : VirtualCamera
    target_frame_xy : (x, y)
        Tracked beacon position in CAMERA-FRAME pixel coordinates.
    dt : float
        Time elapsed this frame (used by camera.move_by for its speed limit).
    gain : float
        Proportional gain (0-1). 1.0 would try to fully correct the error
        in one frame (subject to the camera's max-speed clamp); lower
        values move more gradually.
    """
    error_x, error_y = compute_error(target_frame_xy, camera.image_center())
    command = error_to_command(error_x, error_y)

    # Move the camera in WORLD coordinates by `gain` fraction of the error.
    # Error is in camera-frame pixels, but since our camera-to-world mapping
    # is currently 1:1 (see virtual_camera.py), this maps directly.
    camera.move_by(gain * error_x, gain * error_y, dt)

    error_magnitude = (error_x ** 2 + error_y ** 2) ** 0.5
    return ControlResult(error_x=error_x, error_y=error_y,
                          error_magnitude_px=error_magnitude, command=command)
