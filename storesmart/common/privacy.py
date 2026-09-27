"""Privacy-preserving rendering: blurred view, zero-frame (dots only) view,
and an in-memory people-free background for setup screens.

Nothing in this module writes an image to disk or sends one anywhere — it
only returns arrays held in memory for on-screen display.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np

VIEWS = ("blurred", "zero-frame", "raw")


def render_blurred(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Gaussian-blur each person bounding box region. Returns a new array."""
    img = frame.copy()
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 > 4 and y2 - y1 > 4:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (0, 0), 18)
    return img


def render_zero_frame(frame: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """Black canvas with anonymous dots at foot points. No pixel of the
    original frame is reused."""
    img = np.zeros_like(frame)
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), 7, (255, 200, 0), -1)
    return img


def render_raw(frame: np.ndarray) -> np.ndarray:
    """Unmodified frame — setup only. Callers must overlay a 'RAW VIEW -
    setup only' label; this function does not persist or transmit anything."""
    return frame.copy()


class BackgroundEstimator:
    """Median of recent frames -> a people-free background, kept in memory.
    Used by setup/calibration screens instead of a saved photo."""

    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)

    def add(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def ready(self) -> bool:
        return len(self._frames) >= min(5, self.max_frames)

    def compute(self) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        stack = np.stack(list(self._frames), axis=0)
        return np.median(stack, axis=0).astype(np.uint8)
