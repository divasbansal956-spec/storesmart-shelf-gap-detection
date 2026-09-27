"""Shelf gap detection tuned for demo reliability, not state-of-the-art
accuracy: a plain white backdrop sits behind the products, and each slot's
fill level is estimated from how much backdrop colour is visible.

  - >60% backdrop visible in a slot -> empty
  - 30-60% -> low
  - <30% -> ok
  - a snapshot is skipped entirely if a person overlaps the shelf (so a
    shopper standing in front doesn't get misread as "empty")
  - a state change is only confirmed after 2 consecutive snapshots agree,
    to filter out one-off lighting/occlusion glitches
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SlotState:
    slot: str
    state: str = "ok"       # ok | low | empty
    fill_pct: float = 0.0
    pending_state: str | None = None
    pending_count: int = 0


class GapDetector:
    def __init__(self, slots: list[dict], hsv_lower: list[int], hsv_upper: list[int],
                 empty_threshold_pct: float = 60, low_threshold_pct: float = 30,
                 confirm_frames: int = 2):
        self.slots = slots
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        # A slot may carry its own backdrop band: the surface behind a slot is
        # not always the white wall (wood shelf, painted panel), and one global
        # band can only ever match one of them.
        self.slot_bands: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in slots:
            if "hsv_lower" in s and "hsv_upper" in s:
                self.slot_bands[s["slot"]] = (np.array(s["hsv_lower"], dtype=np.uint8),
                                              np.array(s["hsv_upper"], dtype=np.uint8))
        self.empty_threshold_pct = empty_threshold_pct
        self.low_threshold_pct = low_threshold_pct
        self.confirm_frames = confirm_frames
        self.states: dict[str, SlotState] = {s["slot"]: SlotState(slot=s["slot"]) for s in slots}

    @staticmethod
    def _person_overlaps_shelf(person_boxes: list[tuple[int, int, int, int]], slot: dict) -> bool:
        sx1, sy1, sx2, sy2 = slot["x"], slot["y"], slot["x"] + slot["w"], slot["y"] + slot["h"]
        for x1, y1, x2, y2 in person_boxes:
            ix1, iy1 = max(sx1, x1), max(sy1, y1)
            ix2, iy2 = min(sx2, x2), min(sy2, y2)
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False

    def _backdrop_fraction(self, frame: np.ndarray, slot: dict) -> float:
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower, upper = self.slot_bands.get(slot["slot"], (self.hsv_lower, self.hsv_upper))
        mask = cv2.inRange(hsv, lower, upper)
        return 100.0 * float(np.count_nonzero(mask)) / mask.size

    def update(self, frame: np.ndarray, person_boxes: list[tuple[int, int, int, int]]) -> list[tuple[str, str]]:
        """Returns a list of (slot, new_state) for slots whose confirmed
        state just changed."""
        changes = []
        for slot in self.slots:
            name = slot["slot"]
            st = self.states[name]
            if self._person_overlaps_shelf(person_boxes, slot):
                continue  # skip this snapshot for this slot entirely

            fill_pct = self._backdrop_fraction(frame, slot)
            st.fill_pct = fill_pct
            if fill_pct > self.empty_threshold_pct:
                candidate = "empty"
            elif fill_pct > self.low_threshold_pct:
                candidate = "low"
            else:
                candidate = "ok"

            if candidate == st.pending_state:
                st.pending_count += 1
            else:
                st.pending_state, st.pending_count = candidate, 1

            if st.pending_count >= self.confirm_frames and candidate != st.state:
                st.state = candidate
                changes.append((name, candidate))
        return changes
