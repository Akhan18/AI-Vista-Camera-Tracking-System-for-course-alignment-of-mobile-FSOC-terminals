import unittest
import numpy as np
import cv2

from src.detection.classical_detector import ClassicalDetector


class TestClassicalDetector(unittest.TestCase):

    def setUp(self):
        self.detector = ClassicalDetector(threshold=100, min_area_px=4)

    def _make_frame_with_square(self, width=640, height=480,
                                 square_center=(320, 240), square_size=10,
                                 background=10, brightness=255):
        frame = np.full((height, width, 3), background, dtype=np.uint8)
        half = square_size // 2
        cx, cy = square_center
        top_left = (cx - half, cy - half)
        bottom_right = (cx + half, cy + half)
        cv2.rectangle(frame, top_left, bottom_right, (brightness,) * 3, thickness=-1)
        return frame

    def test_detects_bright_square_at_correct_centroid(self):
        frame = self._make_frame_with_square(square_center=(320, 240))
        result = self.detector.detect(frame)

        self.assertTrue(result["found"])
        cx, cy = result["centroid"]
        self.assertAlmostEqual(cx, 320, delta=1.0)
        self.assertAlmostEqual(cy, 240, delta=1.0)

    def test_no_detection_on_empty_frame(self):
        frame = np.full((480, 640, 3), 10, dtype=np.uint8)  # pure background
        result = self.detector.detect(frame)
        self.assertFalse(result["found"])
        self.assertIsNone(result["centroid"])

    def test_detects_square_at_different_positions(self):
        for pos in [(50, 50), (600, 400), (100, 400)]:
            frame = self._make_frame_with_square(square_center=pos)
            result = self.detector.detect(frame)
            self.assertTrue(result["found"], f"Failed to detect at {pos}")
            cx, cy = result["centroid"]
            self.assertAlmostEqual(cx, pos[0], delta=1.0)
            self.assertAlmostEqual(cy, pos[1], delta=1.0)

    def test_confidence_is_between_zero_and_one(self):
        frame = self._make_frame_with_square(square_center=(320, 240))
        result = self.detector.detect(frame)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
