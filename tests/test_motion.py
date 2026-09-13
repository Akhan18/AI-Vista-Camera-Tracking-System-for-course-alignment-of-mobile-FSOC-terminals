import unittest
import math

from src.utils.config_loader import load_scenario
from src.simulation import motion

BASE = "configs/default.yaml"


class TestMotion(unittest.TestCase):

    def test_straight_line_start_and_end(self):
        cfg = load_scenario(BASE, "configs/straight_line.yaml")
        x0, y0 = cfg.motion.straight_line.start_xy
        x1, y1 = cfg.motion.straight_line.end_xy

        # At t=0, beacon should be exactly at the start point.
        x, y = motion.straight_line(0.0, cfg)
        self.assertAlmostEqual(x, x0)
        self.assertAlmostEqual(y, y0)

        # After traveling the full distance, beacon should be at the end
        # point and should NOT overshoot past it.
        dist = math.hypot(x1 - x0, y1 - y0)
        total_time = dist / cfg.motion.speed_px_per_s
        x, y = motion.straight_line(total_time + 100, cfg)  # well past the end
        self.assertAlmostEqual(x, x1, places=3)
        self.assertAlmostEqual(y, y1, places=3)

    def test_circular_radius_is_constant(self):
        cfg = load_scenario(BASE, "configs/circular.yaml")
        cx, cy = cfg.motion.circular.center_xy
        radius = cfg.motion.circular.radius_px

        for t in [0, 1.3, 5.0, 12.7]:
            x, y = motion.circular(t, cfg)
            dist_from_center = math.hypot(x - cx, y - cy)
            self.assertAlmostEqual(dist_from_center, radius, places=3)

    def test_figure8_stays_within_bounding_box(self):
        cfg = load_scenario(BASE, "configs/figure8.yaml")
        cx, cy = cfg.motion.figure8.center_xy
        w, h = cfg.motion.figure8.width_px, cfg.motion.figure8.height_px

        for t in [0, 1, 2, 3, 4, 5, 10, 20]:
            x, y = motion.figure8(t, cfg)
            self.assertLessEqual(abs(x - cx), w / 2 + 1e-6)
            self.assertLessEqual(abs(y - cy), h / 2 + 1e-6)


if __name__ == "__main__":
    unittest.main()
