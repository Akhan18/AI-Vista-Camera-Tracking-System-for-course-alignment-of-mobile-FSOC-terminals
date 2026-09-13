"""
kalman_tracker.py

Kalman-filter based tracker for the single-beacon coarse-alignment
simulation.

The tracker uses a short acquisition-confirmation period so that a single
successful detection does not immediately count as stable acquisition.
"""

import cv2
import numpy as np


class KalmanTracker:
    def __init__(
        self,
        dt: float,
        max_missed_frames: int = 10,
        acquisition_frames: int = 5,
    ):
        """
        Parameters
        ----------
        dt : float
            Time between simulation frames.

        max_missed_frames : int
            Number of consecutive missed detections tolerated before
            declaring the target LOST.

        acquisition_frames : int
            Number of consecutive successful detections required before
            declaring stable TRACKING.
        """

        self.kf = cv2.KalmanFilter(4, 2)

        # Constant-velocity model:
        # x' = x + vx*dt
        # y' = y + vy*dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Position measurement only.
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # Measurement noise.
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.initialized = False
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames

        # Acquisition confirmation state.
        self.acquisition_frames = acquisition_frames
        self.consecutive_detections = 0
        self.acquired = False

    def init(self, x: float, y: float):
        """Initialize the Kalman filter with the first detected position."""

        self.kf.statePre = np.array(
            [[x], [y], [0], [0]],
            dtype=np.float32,
        )

        self.kf.statePost = np.array(
            [[x], [y], [0], [0]],
            dtype=np.float32,
        )

        self.initialized = True
        self.missed_frames = 0
        self.consecutive_detections = 1
        self.acquired = False

    def update(self, detection: dict) -> dict:
        """
        Advance the tracker by one frame.

        Returns:
            position:
                Current estimated beacon position.

            status:
                "ACQUIRING"
                "TRACKING"
                "PREDICTED"
                "LOST"
                "UNINITIALIZED"
        """

        # ------------------------------------------------------------
        # First detection
        # ------------------------------------------------------------
        if not self.initialized:

            if detection["found"]:
                self.init(*detection["centroid"])

                # First detection starts acquisition confirmation.
                if self.acquisition_frames <= 1:
                    self.acquired = True
                    status = "TRACKING"
                else:
                    status = "ACQUIRING"

                return {
                    "position": detection["centroid"],
                    "status": status,
                }

            return {
                "position": None,
                "status": "UNINITIALIZED",
            }

        # ------------------------------------------------------------
        # Predict next position
        # ------------------------------------------------------------
        predicted = self.kf.predict()

        # ------------------------------------------------------------
        # Successful detection
        # ------------------------------------------------------------
        if detection["found"]:

            measurement = np.array(
                [
                    [detection["centroid"][0]],
                    [detection["centroid"][1]],
                ],
                dtype=np.float32,
            )

            corrected = self.kf.correct(measurement)

            self.missed_frames = 0
            self.consecutive_detections += 1

            # Stable acquisition after enough consecutive detections.
            if not self.acquired:
                if self.consecutive_detections >= self.acquisition_frames:
                    self.acquired = True
                    status = "TRACKING"
                else:
                    status = "ACQUIRING"
            else:
                status = "TRACKING"

            return {
                "position": (
                    float(corrected[0, 0]),
                    float(corrected[1, 0]),
                ),
                "status": status,
            }

        # ------------------------------------------------------------
        # Detection missed
        # ------------------------------------------------------------
        self.missed_frames += 1
        self.consecutive_detections = 0

        if self.missed_frames > self.max_missed_frames:
            self.acquired = False

            return {
                "position": (
                    float(predicted[0, 0]),
                    float(predicted[1, 0]),
                ),
                "status": "LOST",
            }

        # If acquisition has not yet been confirmed, a missed frame
        # means the confirmation sequence must restart.
        if not self.acquired:
            return {
                "position": (
                    float(predicted[0, 0]),
                    float(predicted[1, 0]),
                ),
                "status": "ACQUIRING",
            }

        # Already acquired: temporarily rely on prediction.
        return {
            "position": (
                float(predicted[0, 0]),
                float(predicted[1, 0]),
            ),
            "status": "PREDICTED",
        }