"""Video input: threaded "latest frame" readers for live streams, a snapshot
poller for shelf cameras, and a simulate mode. Frames live in memory only and
are never written to disk from this module.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests


#: cv2 rotation codes by degrees clockwise.
_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 degrees.

    A phone mounted sideways to watch a doorway streams a rotated image, and
    YOLO is trained on upright people — a person lying sideways in frame
    detects far worse. Rotating on input fixes detection everywhere
    downstream, since every module sees the corrected frame.
    """
    code = _ROTATIONS.get(int(degrees) % 360)
    return frame if code is None else cv2.rotate(frame, code)


class FrameSource:
    """Live streams: a background thread always holds only the newest frame,
    so processing never falls behind (no buffer build-up). Video files are
    read frame by frame instead. Reconnects automatically on failure."""

    def __init__(self, src, proc_width: int = 960, reconnect_delay: float = 2.0,
                 rotate: int = 0):
        self.src = src
        self.proc_width = proc_width
        self.reconnect_delay = reconnect_delay
        self.rotate = int(rotate) % 360
        self.is_file = isinstance(src, str) and os.path.isfile(src)
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.connected = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
        self.cap = self._open()
        if not self.is_file:
            threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(int(self.src) if str(self.src).isdigit() else self.src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = cap.isOpened()
        if not self.connected:
            self.last_error = f"cannot reach camera at {self.src} — check the hotspot / URL"
        return cap

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        # rotate first, so proc_width caps the width of the upright image
        frame = rotate_frame(frame, self.rotate)
        h, w = frame.shape[:2]
        if w > self.proc_width:
            frame = cv2.resize(frame, (self.proc_width, int(h * self.proc_width / w)))
        return frame

    def _loop(self) -> None:
        while self.running:
            if not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                self.cap.release()
                self.cap = self._open()
                continue
            ok, frame = self.cap.read()
            if not ok:
                self.connected = False
                self.last_error = f"lost connection to {self.src}, reconnecting..."
                time.sleep(self.reconnect_delay)
                self.cap.release()
                self.cap = self._open()
                continue
            self.connected = True
            with self._lock:
                self.frame = self._resize(frame)

    def read(self) -> Optional[np.ndarray]:
        if self.is_file:
            ok, frame = self.cap.read()
            return self._resize(frame) if ok else None
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.cap.release()


class MjpegStreamReader:
    """Reads an MJPEG-over-HTTP stream with `requests` instead of OpenCV.

    OpenCV delegates http:// to FFmpeg, which is fussy: it fails on some
    phone camera apps, on URL-embedded credentials, and on slow/high-latency
    links, giving only "couldn't read video stream" with no reason. requests
    handles Basic auth and redirects properly and lets us control timeouts,
    so phone cameras are far more dependable this way.

    Frames are decoded in memory and only the newest is kept, matching
    FrameSource's contract. Nothing is ever written to disk.
    """

    _SOI = b"\xff\xd8"  # JPEG start-of-image
    _EOI = b"\xff\xd9"  # JPEG end-of-image
    _MAX_BUFFER = 8 * 1024 * 1024

    def __init__(self, url: str, proc_width: int = 960, rotate: int = 0,
                 timeout_s: float = 15.0, reconnect_delay: float = 2.0):
        self.url = url
        self.proc_width = proc_width
        self.rotate = int(rotate) % 360
        self.timeout_s = timeout_s
        self.reconnect_delay = reconnect_delay
        self.frame: Optional[np.ndarray] = None
        self.running = True
        self.connected = False
        self.last_error: Optional[str] = None
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        frame = rotate_frame(frame, self.rotate)
        h, w = frame.shape[:2]
        if w > self.proc_width:
            frame = cv2.resize(frame, (self.proc_width, int(h * self.proc_width / w)))
        return frame

    @classmethod
    def _take_latest_jpeg(cls, buffer: bytearray) -> Optional[bytes]:
        """Consume complete multipart parts from `buffer`, returning the last
        complete JPEG (or None if no part is complete yet).

        Each part's own Content-Length is used rather than scanning for JPEG
        start/end markers: phone cameras embed EXIF thumbnails, so a frame
        contains *nested* SOI/EOI markers and marker-scanning slices out a
        corrupt image that never decodes.
        """
        latest: Optional[bytes] = None
        while True:
            separator = buffer.find(b"\r\n\r\n")
            if separator == -1:
                break
            headers = bytes(buffer[:separator])
            match = re.search(rb"Content-Length:\s*(\d+)", headers, re.IGNORECASE)
            if match is None:
                del buffer[:separator + 4]  # not a part header we understand
                continue
            body_start = separator + 4
            length = int(match.group(1))
            if len(buffer) < body_start + length:
                break  # rest of this frame hasn't arrived yet
            latest = bytes(buffer[body_start:body_start + length])
            del buffer[:body_start + length]
        return latest

    def _loop(self) -> None:
        while self.running:
            try:
                with requests.get(self.url, stream=True, timeout=self.timeout_s) as resp:
                    resp.raise_for_status()
                    self.connected = True
                    self.last_error = None
                    buffer = bytearray()
                    for chunk in resp.iter_content(chunk_size=16384):
                        if not self.running:
                            return
                        if not chunk:
                            continue
                        buffer.extend(chunk)
                        # Decode only the newest complete frame per chunk batch:
                        # phone cameras can push far more frames than we need,
                        # and this keeps us on the live edge instead of working
                        # through a backlog.
                        jpeg = self._take_latest_jpeg(buffer)
                        if jpeg is not None:
                            decoded = cv2.imdecode(
                                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded is not None:
                                with self._lock:
                                    self.frame = self._prepare(decoded)
                        if len(buffer) > self._MAX_BUFFER:
                            buffer.clear()  # never saw a full frame — don't grow forever
            except Exception as exc:
                self.connected = False
                self.last_error = f"cannot read stream at {self.url} — {exc}"
                time.sleep(self.reconnect_delay)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def release(self) -> None:
        self.running = False

    # FrameSource-compatible alias so callers can treat the two the same
    def stop(self) -> None:
        self.release()


class SnapshotPoller:
    """Polls a still-image URL (e.g. IP Webcam's /shot.jpg) every `interval_s`
    seconds on a background thread. Used for the shelf camera, where a full
    video stream isn't needed."""

    def __init__(self, url: str, interval_s: float = 3.0, timeout_s: float = 5.0,
                 rotate: int = 0):
        self.url = url
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.rotate = int(rotate) % 360
        self.frame: Optional[np.ndarray] = None
        self.last_error: Optional[str] = None
        self.running = True
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self.running:
            try:
                resp = requests.get(self.url, timeout=self.timeout_s)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = rotate_frame(frame, self.rotate)
                    with self._lock:
                        self.frame = frame
                    self.last_error = None
            except Exception as exc:  # network hiccup, phone asleep, wrong URL
                self.last_error = f"cannot reach shelf camera at {self.url} — {exc}"
            time.sleep(self.interval_s)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False


def is_snapshot_url(url: str) -> bool:
    return str(url).lower().split("?")[0].endswith((".jpg", ".jpeg", ".png"))


def open_source(cam_cfg: dict, proc_width: int = 960):
    """Build the right reader for a camera config entry from cameras.yaml.

    http(s) video streams go through MjpegStreamReader rather than OpenCV,
    because FFmpeg is unreliable with phone camera apps (auth, slow links).
    Webcam indices, files and rtsp:// still use FrameSource/OpenCV.
    """
    url = cam_cfg.get("url", "simulate")
    if url == "simulate":
        return None
    rotate = cam_cfg.get("rotate", 0)
    if str(url).lower().startswith(("http://", "https://")) and not is_snapshot_url(url):
        return MjpegStreamReader(url, proc_width=proc_width, rotate=rotate)
    return FrameSource(url, proc_width=proc_width, rotate=rotate)
