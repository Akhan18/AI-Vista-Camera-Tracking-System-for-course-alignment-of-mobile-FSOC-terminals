"""
environment.py

The "world" is the 2D coordinate space the beacon moves around in
(default 2000x2000, from configs/default.yaml -> world.width_px/height_px).

Think of it as a large sky/canvas. The camera (Phase 4) only ever sees a
small window (640x480) of this canvas at a time. This module knows nothing
about the camera, detection, or tracking — it only owns world geometry.
"""

from dataclasses import dataclass


@dataclass
class World:
    width_px: int
    height_px: int

    def clamp_point(self, x: float, y: float) -> tuple[float, float]:
        """Keep a point inside world bounds (used so the beacon/camera
        never wanders off the edge of the simulated space)."""
        cx = min(max(x, 0), self.width_px)
        cy = min(max(y, 0), self.height_px)
        return cx, cy

    @classmethod
    def from_config(cls, cfg) -> "World":
        return cls(width_px=cfg.world.width_px, height_px=cfg.world.height_px)
