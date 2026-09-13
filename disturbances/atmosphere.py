"""
atmosphere.py

Simulates atmospheric/environmental degradations relevant to an
outdoor FSOC terminal: low ambient light, haze, and fog. These reduce
contrast between the beacon and background, stressing the detector's
threshold-based approach.
"""

import numpy as np


def apply_low_light(frame: np.ndarray, brightness_factor: float = 0.3) -> np.ndarray:
    """Scale overall frame brightness down (simulates dusk/dawn/overcast)."""
    dimmed = frame.astype(np.float32) * brightness_factor
    return np.clip(dimmed, 0, 255).astype(np.uint8)


def apply_haze(frame: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Blend the frame with a uniform gray veil, reducing contrast --
    a simplified atmospheric scattering model (no depth information
    available in this 2D simulation, so we use a single global veil).
    """
    veil = np.full_like(frame, 180)  # light gray haze color
    blended = (1 - intensity) * frame.astype(np.float32) + intensity * veil.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_fog(frame: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Similar to haze but denser/whiter and slightly blurred, approximating
    thicker fog. Kept as a distinct, tunable effect from haze.
    """
    import cv2
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    veil = np.full_like(frame, 220)  # near-white fog color
    blended = (1 - intensity) * blurred.astype(np.float32) + intensity * veil.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
