# Shelf Gap Detection & Stock Alerts

A real-time computer-vision pipeline for retail shelf monitoring: a camera
watches shelf slots, detects when a product is missing, and cross-references
a stock database to trigger refill, reorder, or shrinkage-mismatch alerts.
Designed for constrained edge hardware (Raspberry Pi class devices,
targeting a Qualcomm QCS6490 NPU in production) rather than sending video to
the cloud — no frame is ever written to disk or transmitted, only small JSON
events leave the pipeline.

This is Phase 3 of **StoreSmart**, a 5-person team project for Smart India
Hackathon. Full team repo: https://github.com/YuvrajDeol/StoreSmart

## My contribution

The base pipeline — the HSV empty/low/ok detector, two-frame temporal
confirmation, person-occlusion skip, and the stock rules engine that turns a
confirmed empty slot into a refill/reorder/mismatch alert — was built by
teammate [Yuvraj](https://github.com/YuvrajDeol) as the initial Phase 3
implementation. I built the calibration layer on top of it:

- **Auto-calibration tool** (`storesmart/dashboard/pages/7_Shelf_Calibration.py`,
  new) — a Streamlit page that finds shelf slot boundaries automatically
  instead of requiring hand-typed coordinates: click-detect objects in a
  single live frame (the frame's median colour vs. each region's distance
  from it in CIE Lab space), or show it an empty vs. a stocked shelf and let
  it diff the two.
- **Per-slot backdrop bands** (`gap_detector.py`) — diagnosed and fixed a
  real bug: the original detector used one global HSV colour band shared by
  every slot, which only works if every slot sits against the same surface.
  A slot on a wooden shelf could never report empty, because its band was
  fitted to a white wall. Slots may now carry their own band.
- **Shadow-resistant band fitting** — the naive fit samples the ring of
  backdrop around a box, which a shadow under a product can drag down until
  the product itself reads as "empty shelf." Built a fitter that scores
  candidate bounds against both the backdrop and the product population,
  cutting false-positive misreads by more than half during testing on a
  live camera.
- **Dependency reduction** — added a `--no-person-tracker` flag so the
  pipeline runs without `ultralytics` installed.

## What the full pipeline does

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

## Why HSV, not a neural network

For a fixed rectangle answering "is something here or not", a lightweight
colour-space test is more reliable and runs in milliseconds on constrained
edge hardware — the deployment target for this project is a Qualcomm
QCS6490 NPU. A detection network is used elsewhere in the project (person
detection, to avoid misreading a slot while a shopper stands in front of
it), but not for this per-slot check.

## Privacy by design

A project-wide guarantee, not specific to my contribution: no camera frame
is ever written to disk or transmitted. Frames exist only in memory inside
the detector; the only thing that leaves it is a small validated JSON event
(`{"slot": "A2", "fill_pct": 84.2, "state": "empty"}`). Enforced by a
repo-wide test that fails the build if an image-write call is ever added to
this code path.

## Structure

```
storesmart/phase3_shelf/
  gap_detector.py     HSV band -> fill% -> ok / low / empty (Yuvraj); per-slot
                       backdrop bands (me)
  run.py              polling loop (Yuvraj); --no-person-tracker flag (me)
  slots.py            slot definitions (Yuvraj)

storesmart/stock/
  rules.py            refill / reorder / mismatch decision logic (Yuvraj)
  forecast.py         sales-average run-out forecasting (Yuvraj)
  db.py, seed.py       SQLite schema and demo data (Yuvraj)

storesmart/dashboard/pages/7_Shelf_Calibration.py
                       calibration UI: click-to-detect slots, auto-fit each
                       slot's backdrop colour range, live preview (me, new)

storesmart/common/
  video.py, detector.py, bus.py, events.py, privacy.py
                       shared camera polling, event validation/bus, and
                       privacy primitives used across all four phases of
                       the project (not authored by me — included here so
                       the module above runs standalone)
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
[full StoreSmart repo](https://github.com/YuvrajDeol/StoreSmart) —
`git log --follow` on any file there shows exactly who wrote what.*
