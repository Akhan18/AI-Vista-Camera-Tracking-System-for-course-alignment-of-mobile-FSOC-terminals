"""
noise.py

Sensor/channel noise models applied to a rendered frame before it reaches
the detector. Used to measure how robust detection/tracking is under
degraded conditions (Phase 13 benchmarking).
"""

import numpy as np


def add_gaussian_noise(frame: np.ndarray, mean: float = 0, stddev: float = 15) -> np.ndarray:
    """Add zero-mean Gaussian sensor noise (models thermal/read noise)."""
    noise = np.random.normal(mean, stddev, frame.shape).astype(np.float32)
    noisy = frame.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(frame: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """
    Randomly sets a fraction of pixels to pure white ("salt") or pure black
    ("pepper"). Models sensor defects / transmission glitches.

    Parameters
    ----------
    amount : float
        Fraction of pixels affected in total (half salt, half pepper).
    """
    noisy = frame.copy()
    h, w = frame.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white)
    ys = np.random.randint(0, h, num_pixels // 2)
    xs = np.random.randint(0, w, num_pixels // 2)
    noisy[ys, xs] = 255

    # Pepper (black)
    ys = np.random.randint(0, h, num_pixels // 2)
    xs = np.random.randint(0, w, num_pixels // 2)
    noisy[ys, xs] = 0

    return noisy


def add_poisson_noise(frame: np.ndarray) -> np.ndarray:
    """
    Poisson (shot) noise: models photon-counting statistics -- noise
    magnitude scales with signal intensity itself, unlike Gaussian noise.
    """
    # Scale down, apply Poisson, scale back -- standard approach for
    # simulating shot noise on 8-bit images.
    scaled = frame.astype(np.float32) / 255.0 * 30  # arbitrary photon-count scale
    noisy = np.random.poisson(scaled).astype(np.float32) / 30 * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)
