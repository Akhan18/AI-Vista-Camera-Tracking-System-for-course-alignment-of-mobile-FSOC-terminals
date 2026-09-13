"""
target.py

The Beacon class represents the moving target the whole system is trying
to detect, track, and centre. It owns its appearance (size/color) and
delegates "where am I at time t" to the motion functions in motion.py.
"""

from src.simulation import motion


class Beacon:
    def __init__(self, cfg):
        self.cfg = cfg
        self.width = cfg.beacon.size_px.width
        self.height = cfg.beacon.size_px.height
        self.color_bgr = tuple(cfg.beacon.color_bgr)
        self.motion_type = cfg.motion.type
        self._random_state: dict = {}  # only used for random_walk motion

    def position_at(self, t: float, dt: float) -> tuple[float, float]:
        """Return the beacon's (x, y) centre in world coordinates at time t."""
        if self.motion_type == "random":
            return motion.random_walk(t, self.cfg, dt, self._random_state)

        motion_fn = motion.MOTION_FUNCTIONS.get(self.motion_type)
        if motion_fn is None:
            raise ValueError(
                f"Unknown motion.type '{self.motion_type}'. "
                f"Supported: {list(motion.MOTION_FUNCTIONS.keys()) + ['random']}"
            )
        return motion_fn(t, self.cfg)
