# Shelf Gap Detection & Stock Alerts

An edge-deployable computer-vision pipeline for retail shelf monitoring:
a camera watches shelf slots, detects when a product is missing, and
cross-references a stock database to tell a store manager exactly what to
do about it — refill from the storeroom, reorder from the distributor, or
investigate a shrinkage/stock mismatch. Designed to run on constrained edge
hardware (Raspberry Pi class devices, targeting a Qualcomm QCS6490 NPU in
production) rather than sending video to the cloud — no frame is ever
written to disk or transmitted; only small JSON events leave the pipeline.

Prototyped and demonstrated using an Android phone in IP-camera mode as a
stand-in for a mounted shelf camera, polled over the local network exactly
as a fixed edge camera would be — swapping in real camera hardware requires
no change to the detection or alerting code below.

Built as my independent module (Phase 3 of 4) within **StoreSmart**, a
5-person hackathon project for Smart India Hackathon. Full team repo:
https://github.com/YuvrajDeol/StoreSmart

## What it does

- Polls a shelf snapshot from an IP camera every few seconds
- For each calibrated slot, measures how much of it still looks like bare
  shelf (an HSV colour-range test) — the resulting percentage says whether
  the slot is `ok`, `low`, or `empty`
- Two-frame confirmation filters out false triggers from a shopper's hand or
  arm passing through
- On a confirmed empty slot, cross-references a SQLite stock database and
  raises up to three alerts:
  - **Refill** — stock exists in the storeroom, go get it
  - **Reorder** — projected to run out before the next supplier visit, based
    on a 14-day sales average and the supplier's lead time
  - **Mismatch** — the database says stock is on the shelf and nothing has
    sold recently, but the camera disagrees (shrinkage / misplaced item)
- A calibration tool (built with Streamlit) removes the need to hand-type
  slot coordinates: click-detect objects in a live frame, or show it an
  empty vs. stocked shelf and let it find the difference. Each slot fits its
  own backdrop colour range, so slots against different surfaces (a white
  wall, a wooden shelf) work side by side without manual tuning per surface.

## Why HSV, not a neural network

For a fixed rectangle answering "is something here or not", a lightweight
colour-space test is more reliable and runs in milliseconds on constrained
edge hardware — the deployment target for this project is a Qualcomm
QCS6490 NPU. A detection network is used elsewhere in the project (person
detection, to avoid misreading a slot while a shopper stands in front of
it), but not for this per-slot check.

## Privacy by design

No camera frame is ever written to disk or transmitted. Frames exist only
in memory inside the detector; the only thing that leaves it is a small
validated JSON event (`{"slot": "A2", "fill_pct": 84.2, "state": "empty"}`).
Enforced by a repo-wide test that fails the build if an image-write call is
ever added to this code path.

## Structure

```
storesmart/phase3_shelf/
  gap_detector.py     HSV band -> fill% -> ok / low / empty, with per-slot
                       backdrop bands
  run.py              polling loop: reads the camera, runs detection,
                       emits events, calls the stock rules
  slots.py            slot definitions (rectangle + linked stock item)

storesmart/stock/
  rules.py            refill / reorder / mismatch decision logic
  forecast.py         sales-average run-out forecasting
  db.py, seed.py       SQLite schema and demo data

storesmart/dashboard/pages/7_Shelf_Calibration.py
                       calibration UI: click-to-detect slots, auto-fit each
                       slot's backdrop colour range, live preview

storesmart/common/
  video.py, detector.py, bus.py, events.py, privacy.py
                       shared camera polling, event validation/bus, and
                       privacy primitives used across all four phases of
                       the project (not authored solely by me — included
                       here so the module above runs standalone)
```

## Running it

```bash
pip install -r requirements.txt
python -m storesmart.stock.seed                 # seed demo stock data
python -m storesmart.phase3_shelf.run --no-person-tracker --headless
streamlit run storesmart/dashboard/app.py       # in a second terminal
```

You'll also need `config/cameras.local.yaml` pointing at an IP-camera
snapshot URL (e.g. the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
Android app), and a calibrated `config/shelf_slots.json` — generate the
second one from the Shelf Calibration dashboard page rather than by hand.

---
*Extracted from the team repo for portfolio purposes. Commit history for
this module lives on branch `feat/phase3-shelf-auto-calibration` of the
[full StoreSmart repo](https://github.com/YuvrajDeol/StoreSmart).*
