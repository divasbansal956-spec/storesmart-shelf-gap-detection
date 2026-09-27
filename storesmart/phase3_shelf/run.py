#!/usr/bin/env python3
"""Phase 3: shelf gaps + stock system.

Polls a shelf snapshot every few seconds, detects empty/low slots, and
raises refill/reorder/mismatch alerts by combining that with the stock DB.

Usage:
  python -m storesmart.phase3_shelf.run --simulate
  python -m storesmart.phase3_shelf.run --headless
"""
from __future__ import annotations

import argparse
import time

from storesmart.common.bus import EventBus
from storesmart.common.config import load_cameras, load_settings
from storesmart.phase3_shelf.gap_detector import GapDetector
from storesmart.phase3_shelf.slots import load_slots
from storesmart.sim.shelf_sim import ShelfSimulator
from storesmart.stock.db import get_connection
from storesmart.stock.rules import evaluate_shelf_status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between snapshots")
    ap.add_argument("--no-person-tracker", action="store_true",
                    help="skip YOLO person detection (no ultralytics needed). Slots blocked by a "
                         "shopper will no longer be skipped, so only use this when nobody stands "
                         "between the camera and the shelf.")
    args = ap.parse_args()

    settings = load_settings().get("shelf", {})
    alert_window = load_settings().get("alerts", {}).get("dedupe_window_min", 15)
    slots_cfg = load_slots()
    bus = EventBus()
    conn = get_connection()

    cameras = load_cameras()
    simulate = args.simulate or cameras.get("shelf", {}).get("url", "simulate") == "simulate"

    detector = GapDetector(
        slots=slots_cfg["slots"],
        hsv_lower=slots_cfg["backdrop_hsv_lower"],
        hsv_upper=slots_cfg["backdrop_hsv_upper"],
        empty_threshold_pct=settings.get("empty_threshold_pct", 60),
        low_threshold_pct=settings.get("low_threshold_pct", 30),
        confirm_frames=settings.get("confirm_frames", 2),
    )

    if simulate:
        sim = ShelfSimulator(slots_cfg["slots"])
        source = None
        tracker = None
    else:
        from storesmart.common.video import SnapshotPoller

        cam_cfg = cameras.get("shelf", {})
        source = SnapshotPoller(cam_cfg["url"], interval_s=cam_cfg.get("snapshot_interval_s", args.interval))
        if args.no_person_tracker:
            tracker = None
            print("Person tracker disabled: slots blocked by a shopper will not be skipped.")
        else:
            from storesmart.common.detector import PersonTracker

            tracker = PersonTracker()

    t0 = time.time()
    print("Phase 3 shelf monitor running" + (" (simulate)" if simulate else ""))
    while True:
        now = time.time() - t0
        if simulate:
            frame, person_boxes = sim.step(args.interval)
        else:
            frame = source.read()
            if frame is None:
                if source.last_error:
                    print(source.last_error)
                time.sleep(args.interval)
                continue
            if tracker is None:
                person_boxes = []
            else:
                tracks = tracker.update(frame)
                person_boxes = [box for _, box in tracks]

        changes = detector.update(frame, person_boxes)
        for slot_name, new_state in changes:
            fill_pct = detector.states[slot_name].fill_pct
            bus.emit({"cam": "shelf", "type": "shelf_status", "slot": slot_name,
                       "fill_pct": round(fill_pct, 1), "state": new_state})
            evaluate_shelf_status(conn, bus, slot_name, new_state, dedupe_window_min=alert_window)

        if args.duration and now >= args.duration:
            break
        if args.headless or simulate:
            time.sleep(args.interval)

    print("Phase 3 shelf monitor stopped.")


if __name__ == "__main__":
    main()
