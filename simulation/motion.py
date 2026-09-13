"""
motion.py

Each function here answers one question: "given elapsed time t (seconds),
where is the beacon in world coordinates?"

Keeping motion as pure functions of time (not accumulated step-by-step
state) means: (a) it's trivially testable, (b) restarting/seeking is exact,
no drift, (c) it's easy to reason about and explain to judges.

All functions return (x, y) in world pixel coordinates.
"""

import math
import random


def straight_line(t: float, cfg) -> tuple[float, float]:
    """Beacon moves at constant speed from start_xy to end_xy, then stops."""
    m = cfg.motion.straight_line
    x0, y0 = m.start_xy
    x1, y1 = m.end_xy
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist == 0:
        return x0, y0
    total_time = dist / cfg.motion.speed_px_per_s
    frac = min(t / total_time, 1.0) if total_time > 0 else 1.0
    x = x0 + (x1 - x0) * frac
    y = y0 + (y1 - y0) * frac
    return x, y


def circular(t: float, cfg) -> tuple[float, float]:
    """Beacon moves on a circle at constant angular speed."""
    m = cfg.motion.circular
    cx, cy = m.center_xy
    angle_rad = math.radians(m.angular_speed_deg_per_s * t)
    x = cx + m.radius_px * math.cos(angle_rad)
    y = cy + m.radius_px * math.sin(angle_rad)
    return x, y


def figure8(t: float, cfg) -> tuple[float, float]:
    """Beacon traces a figure-8 (Lissajous curve, freq ratio 1:2)."""
    m = cfg.motion.figure8
    cx, cy = m.center_xy
    angle_rad = math.radians(m.angular_speed_deg_per_s * t)
    x = cx + (m.width_px / 2) * math.sin(angle_rad)
    y = cy + (m.height_px / 2) * math.sin(2 * angle_rad)
    return x, y


def random_walk(t: float, cfg, dt: float, state: dict) -> tuple[float, float]:
    """
    Beacon takes a small random step each call. Unlike the other generators,
    this one is NOT a pure function of t alone -- it needs to remember its
    last position, so the caller passes in a mutable `state` dict that
    holds {"x":, "y":, "rng":}. This is intentional and documented, rather
    than silently making random_walk behave differently from the others.
    """
    m = cfg.motion.random
    if "rng" not in state:
        state["rng"] = random.Random(m.seed)
        state["x"], state["y"] = cfg.world.width_px / 2, cfg.world.height_px / 2

    rng = state["rng"]
    step = m.step_size_px
    state["x"] += rng.uniform(-step, step)
    state["y"] += rng.uniform(-step, step)
    state["x"] = min(max(state["x"], 0), cfg.world.width_px)
    state["y"] = min(max(state["y"], 0), cfg.world.height_px)
    return state["x"], state["y"]


# Registry mapping config's motion.type string to the right function.
# random_walk is handled separately in target.py because of its extra state.
MOTION_FUNCTIONS = {
    "straight_line": straight_line,
    "circular": circular,
    "figure8": figure8,
}
