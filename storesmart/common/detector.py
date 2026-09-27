"""YOLO + ByteTrack wrapper: person detection and motion-only tracking.

Only the person class is detected. Tracking associates boxes across frames by
motion (ByteTrack) — no appearance embeddings, no face detection, no
re-identification. Returns (track_id, bbox) tuples only; frames never leave
this module.
"""
from __future__ import annotations

from typing import Optional


class PersonTracker:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35,
                 imgsz: int = 640, device: Optional[str] = None, tracker: str = "config/bytetrack.yaml"):
        from ultralytics import YOLO  # local import: --simulate needs no torch/ultralytics
        import os

        if not os.path.exists(tracker):
            tracker = "bytetrack.yaml"  # fall back to Ultralytics' bundled default
        self.model = YOLO(model_path)
        self.device = device or self._auto_device()
        self.track_kwargs = dict(
            persist=True, classes=[0], tracker=tracker,
            conf=conf, imgsz=imgsz, verbose=False, device=self.device,
        )

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def update(self, frame) -> list[tuple[int, tuple[int, int, int, int]]]:
        result = self.model.track(frame, **self.track_kwargs)[0]
        out = []
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = (int(v) for v in box)
                out.append((tid, (x1, y1, x2, y2)))
        return out
