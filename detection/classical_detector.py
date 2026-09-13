"""
classical_detector.py

Detects a bright beacon on a dark background using classical computer
vision: denoising + intensity thresholding + connected-component analysis.

WHY CLASSICAL CV FIRST (not YOLO):
The beacon is a small, simple, high-contrast bright spot -- not a complex
object with varied appearance/pose. A thresholding approach is:
  - Extremely fast (no GPU, no model loading).
  - Deterministic and easy to explain to judges.
  - A legitimate technique for bright-spot detection problems such as
    star trackers and optical beacon acquisition systems.

The detector is intentionally kept lightweight. A learned detector can
still be evaluated later under difficult disturbances.
"""

import cv2
import numpy as np


class ClassicalDetector:
    def __init__(
        self,
        threshold: int = 100,
        min_area_px: int = 4,
        use_otsu: bool = False,
    ):
        """
        Parameters
        ----------
        threshold : int
            Grayscale intensity above which a pixel is considered bright
            when fixed thresholding is used.

        min_area_px : int
            Smallest blob area accepted as a valid detection.

        use_otsu : bool
            If True, automatically determine the threshold using Otsu's
            method instead of using the fixed threshold value.
        """

        self.threshold = threshold
        self.min_area_px = min_area_px
        self.use_otsu = use_otsu

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        Run detection on a single frame.

        Returns
        -------
        dict
            {
                "found": bool,
                "centroid": (x, y) or None,
                "area_px": int,
                "confidence": float,
                "threshold_used": float
            }
        """

        # ------------------------------------------------------------
        # Convert image to grayscale
        # ------------------------------------------------------------

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY
        )

        # ------------------------------------------------------------
        # Basic image statistics
        # ------------------------------------------------------------

        gray_max = int(gray.max())
        gray_mean = float(gray.mean())

        # ------------------------------------------------------------
        # Reject an obviously dark frame
        # ------------------------------------------------------------
        #
        # This is important for the target-loss scenario.
        # When the beacon is intentionally hidden, the simulation
        # produces a dark frame. The detector must return found=False.
        #
        if gray_max < self.threshold:
            return {
                "found": False,
                "centroid": None,
                "area_px": 0,
                "confidence": 0.0,
                "threshold_used": float(self.threshold),
            }

        # ------------------------------------------------------------
        # Denoising
        # ------------------------------------------------------------
        #
        # Gaussian noise can create isolated bright pixels and small
        # false blobs. A small Gaussian blur suppresses those fluctuations
        # while preserving the beacon because the beacon is larger than
        # a single pixel.
        #
        filtered = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ------------------------------------------------------------
        # Threshold the filtered image
        # ------------------------------------------------------------

        if self.use_otsu:

            otsu_threshold, binary = cv2.threshold(
                filtered,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            threshold_used = otsu_threshold

        else:

            _, binary = cv2.threshold(
                filtered,
                self.threshold,
                255,
                cv2.THRESH_BINARY,
            )

            threshold_used = self.threshold

        # ------------------------------------------------------------
        # Connected-component analysis
        # ------------------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        if num_labels <= 1:

            return {
                "found": False,
                "centroid": None,
                "area_px": 0,
                "confidence": 0.0,
                "threshold_used": float(threshold_used),
            }

        # ------------------------------------------------------------
        # Select the largest bright blob
        # ------------------------------------------------------------

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        best_idx = np.argmax(areas) + 1

        best_area = int(
            stats[
                best_idx,
                cv2.CC_STAT_AREA
            ]
        )

        # Reject blobs that are too small.
        if best_area < self.min_area_px:

            return {
                "found": False,
                "centroid": None,
                "area_px": best_area,
                "confidence": 0.0,
                "threshold_used": float(threshold_used),
            }

        # ------------------------------------------------------------
        # Beacon candidate geometry
        # ------------------------------------------------------------

        cx, cy = centroids[best_idx]

        blob_width = int(
            stats[
                best_idx,
                cv2.CC_STAT_WIDTH
            ]
        )

        blob_height = int(
            stats[
                best_idx,
                cv2.CC_STAT_HEIGHT
            ]
        )

        # ------------------------------------------------------------
        # Reject unrealistically large blobs
        # ------------------------------------------------------------
        #
        # The beacon is a small target. A very large bright region is
        # more likely to be an image artefact or background illumination
        # than the simulated beacon.
        #
        max_blob_width = frame_bgr.shape[1] // 2
        max_blob_height = frame_bgr.shape[0] // 2

        if (
            blob_width > max_blob_width
            or blob_height > max_blob_height
        ):

            return {
                "found": False,
                "centroid": None,
                "area_px": best_area,
                "confidence": 0.0,
                "threshold_used": float(threshold_used),
            }

        # ------------------------------------------------------------
        # Calculate brightness and contrast
        # ------------------------------------------------------------

        mask = labels == best_idx

        mean_brightness = float(
            gray[mask].mean()
        )

        background_mask = ~mask

        if np.any(background_mask):

            background_brightness = float(
                gray[background_mask].mean()
            )

        else:

            background_brightness = gray_mean

        contrast = (
            mean_brightness
            - background_brightness
        )

        # ------------------------------------------------------------
        # Contrast validation
        # ------------------------------------------------------------
        #
        # The beacon should be brighter than its surroundings.
        #
        # Gaussian noise reduces the apparent contrast, so we use a
        # moderate threshold rather than an overly strict requirement.
        #
        min_contrast = 10.0

        if contrast < min_contrast:

            return {
                "found": False,
                "centroid": None,
                "area_px": best_area,
                "confidence": 0.0,
                "threshold_used": float(threshold_used),
            }

        # ------------------------------------------------------------
        # Detection confidence
        # ------------------------------------------------------------

        confidence = min(
            mean_brightness / 255.0,
            1.0,
        )

        # ------------------------------------------------------------
        # Return successful detection
        # ------------------------------------------------------------

        return {
            "found": True,
            "centroid": (
                float(cx),
                float(cy)
            ),
            "area_px": best_area,
            "confidence": confidence,
            "threshold_used": float(threshold_used),
        }