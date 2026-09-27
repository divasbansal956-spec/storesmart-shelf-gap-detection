"""Shelf calibration tool: connect to the real shelf camera, grab an in-memory
frame (never saved to disk), click two corners to define a slot box, auto-tune
the backdrop HSV range from the wall around that box, preview live fill% per
slot, and save to config/shelf_slots.json + the shelf: section of settings.yaml.

Nothing here writes a frame to disk — st.image renders the in-memory array
straight to the browser. Only coordinates, HSV numbers and thresholds are saved.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from storesmart.common.config import CONFIG_DIR, load_cameras, load_settings, load_yaml, save_yaml
from storesmart.common.privacy import BackgroundEstimator
from storesmart.phase3_shelf.gap_detector import GapDetector
from storesmart.phase3_shelf.slots import load_slots, save_slots
from storesmart.stock.db import get_connection, get_items

st.set_page_config(page_title="StoreSmart — Shelf Calibration", page_icon="🛠️", layout="wide")
st.title("Shelf Calibration")
st.caption(
    "Setup-only screen. Frames are held in memory and rendered straight to the "
    "browser — never saved to disk. Only coordinates and thresholds are written."
)

SLOT_COLUMNS = ["slot", "item_id", "x", "y", "w", "h"]

SLOTS_PATH = CONFIG_DIR / "shelf_slots.json"


def _load_calibration_into_session() -> int:
    """Pull slots, per-slot bands and the global band from disk into session."""
    cfg = load_slots()
    st.session_state.slots_list = [
        {"slot": s["slot"], "item_id": s.get("item_id", 0), "x": s["x"], "y": s["y"],
         "w": s["w"], "h": s["h"]}
        for s in cfg["slots"]
    ]
    # Per-slot backdrop bands, keyed by slot name. Kept beside the editable
    # table rather than in it so hand-editing coordinates can't drop them.
    st.session_state.slot_bands = {
        s["slot"]: ([int(v) for v in s["hsv_lower"]], [int(v) for v in s["hsv_upper"]])
        for s in cfg["slots"] if "hsv_lower" in s and "hsv_upper" in s
    }
    st.session_state.hsv_lower = [int(v) for v in cfg["backdrop_hsv_lower"]]
    st.session_state.hsv_upper = [int(v) for v in cfg["backdrop_hsv_upper"]]
    return len(st.session_state.slots_list)


if "frame" not in st.session_state:
    st.session_state.frame = None
    st.session_state.ui_version = 0
    st.session_state.corner_pts = []
    st.session_state.corner_last_click = None
    st.session_state.empty_ref = None
    st.session_state.slots_mtime = None
    _load_calibration_into_session()

# Session state outlives both a code change and an edit to the config file, so a
# band fitted by an older version of this page keeps driving the preview while
# the monitor — which reads the file every run — disagrees. Re-sync whenever the
# file on disk changes underneath us.
_mtime = SLOTS_PATH.stat().st_mtime if SLOTS_PATH.exists() else None
if _mtime is not None and _mtime != st.session_state.get("slots_mtime"):
    _load_calibration_into_session()
    st.session_state.slots_mtime = _mtime
    st.session_state.ui_version += 1


def grab_frame(url: str) -> np.ndarray | None:
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _frac_in_band(pixels: np.ndarray, lower, upper) -> float:
    if pixels is None or pixels.size == 0:
        return 0.0
    inside = np.all((pixels >= np.array(lower)) & (pixels <= np.array(upper)), axis=1)
    return float(inside.mean()) if inside.size else 0.0


def _subsample(px: np.ndarray, n: int = 20000) -> np.ndarray:
    if px is None or len(px) <= n:
        return px
    idx = np.random.default_rng(0).choice(len(px), n, replace=False)
    return px[idx]


def _refine_band(backdrop_px: np.ndarray, object_px: np.ndarray, lower, upper,
                 free_hue: bool = True, passes: int = 3):
    """Tighten a band, one channel at a time, to keep backdrop pixels in and
    push object pixels out.

    Fitting purely from percentiles of the surrounding backdrop fails whenever
    that ring is contaminated — a shadow under the product, or a darker patch
    of wall, drags the lower V bound down until a black product reads as
    backdrop and a full slot reports 'empty'. Symmetric percentile pairs can't
    escape that, because the contamination is one-sided. So: score candidate
    bounds against both populations and keep whichever actually separates."""
    b, o = _subsample(backdrop_px), _subsample(object_px)
    if b is None or b.size == 0:
        return lower, upper
    lower, upper = list(lower), list(upper)

    def score(lo, hi):
        return _frac_in_band(b, lo, hi) - _frac_in_band(o, lo, hi)

    lo_cands = [0, 0.5, 1, 2, 5, 10, 15, 20, 25, 30]
    hi_cands = [70, 75, 80, 85, 90, 95, 98, 99, 99.5, 100]
    best = score(lower, upper)
    for _ in range(passes):
        improved = False
        for ch in range(3):
            if ch == 0 and free_hue:
                continue  # hue deliberately left wide open on a pale surface
            for cands, slot in ((lo_cands, "lo"), (hi_cands, "hi")):
                for p in cands:
                    trial_lo, trial_hi = list(lower), list(upper)
                    val = int(np.percentile(b[:, ch], p))
                    if slot == "lo":
                        trial_lo[ch] = max(0, val - 2)
                    else:
                        trial_hi[ch] = min(255 if ch else 180, val + 2)
                    if trial_lo[ch] >= trial_hi[ch]:
                        continue
                    s = score(trial_lo, trial_hi)
                    if s > best + 1e-4:
                        best, lower, upper, improved = s, trial_lo, trial_hi, True
        if not improved:
            break
    return lower, upper


def fit_band_from_empty(empty_ref: np.ndarray, rect: dict, stocked: np.ndarray | None = None):
    """Fit an HSV band to how this exact rectangle looks when the slot is
    EMPTY. Returns (lower, upper, empty_match, object_match).

    This is the reliable direction: sampling the ring *around* a box assumes
    the surface behind the product is the same as the surface beside it, which
    is false the moment a slot sits on a wooden shelf instead of the white
    wall. Sampling the slot's own empty pixels makes 'empty' match by
    construction, whatever that surface happens to be."""
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    roi = empty_ref[y:y + h, x:x + w]
    if roi.size == 0:
        return None
    empty_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    obj_hsv = None
    if stocked is not None:
        obj_roi = stocked[y:y + h, x:x + w]
        if obj_roi.size:
            obj_hsv = cv2.cvtColor(obj_roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    ignore_hue = empty_hsv[:, 1].mean() < 45  # grey/white surface: hue is noise
    best = None
    for lo_p, hi_p, pad in ((1, 99, (5, 14, 16)), (2, 98, (4, 12, 14)),
                            (5, 95, (3, 10, 12)), (10, 90, (3, 8, 10))):
        lower = np.clip(np.percentile(empty_hsv, lo_p, axis=0) - pad, 0, [180, 255, 255])
        upper = np.clip(np.percentile(empty_hsv, hi_p, axis=0) + pad, 0, [180, 255, 255])
        if ignore_hue:
            lower[0], upper[0] = 0, 180
        lower, upper = lower.astype(int).tolist(), upper.astype(int).tolist()
        if obj_hsv is not None:
            lower, upper = _refine_band(empty_hsv, obj_hsv, lower, upper, free_hue=ignore_hue)
        empty_match = _frac_in_band(empty_hsv, lower, upper)
        object_match = _frac_in_band(obj_hsv, lower, upper) if obj_hsv is not None else 0.0
        if best is None or (empty_match - object_match) > (best[2] - best[3]):
            best = (lower, upper, empty_match, object_match)
        if empty_match > 0.9 and object_match < 0.25:
            break
    return best


def _boxes_from_mask(mask: np.ndarray, min_area_frac: float, max_boxes: int,
                     pad_frac: float, pad_down_frac: float) -> list[dict]:
    """Contours -> padded, in-bounds, left-to-right slot rectangles."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = mask.shape
    min_area = min_area_frac * img_h * img_w
    found = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        px, py = int(bw * pad_frac), int(bh * pad_frac)
        nx1, ny1 = max(0, bx - px), max(0, by - py)
        nx2 = min(img_w, bx + bw + px)
        ny2 = min(img_h, by + bh + py + int(bh * pad_down_frac))
        found.append(((nx2 - nx1) * (ny2 - ny1),
                      {"x": nx1, "y": ny1, "w": nx2 - nx1, "h": ny2 - ny1}))
    found.sort(key=lambda t: -t[0])
    return sorted((r for _, r in found[:max_boxes]), key=lambda r: r["x"])


def detect_objects_here(frame: np.ndarray, min_area_frac: float = 0.0015,
                        max_boxes: int = 8, pad_frac: float = 0.15,
                        pad_down_frac: float = 0.25) -> list[dict]:
    """Find products in a SINGLE frame — no empty-shelf reference needed.

    The backdrop is whatever colour covers most of the frame, so the frame's
    median colour is a robust estimate of it (robust in the literal sense: the
    products would have to outweigh the backdrop by area to shift a median).
    Anything perceptually far from that colour is a product. Distance is
    measured in CIE Lab, where Euclidean distance tracks how different two
    colours actually look — in HSV a dark object and a bright backdrop can sit
    numerically close on hue while looking nothing alike."""
    lab = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), (7, 7), 0)
    backdrop = np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - backdrop, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=3)
    return _boxes_from_mask(mask, min_area_frac, max_boxes, pad_frac, pad_down_frac)


def detect_objects(empty_ref: np.ndarray, stocked: np.ndarray,
                   min_area_frac: float = 0.0006, max_boxes: int = 8,
                   pad_frac: float = 0.15, pad_down_frac: float = 0.25) -> list[dict]:
    """Bounding boxes of whatever appeared between the empty and stocked
    frames. Backdrop-agnostic: it keys off change, not colour, so it works on
    a wooden shelf as happily as on the white wall. The camera must not have
    moved between the two grabs.

    The raw contour hugs the product, but a slot is the patch of shelf the
    product is *allowed to sit in*, not the product's silhouette — a box drawn
    tight around a cup reads 'empty' the moment someone sets the cup back
    slightly lower. `pad_frac` grows the box on all sides and `pad_down_frac`
    extends it further toward the shelf surface, where restocked items land."""
    if empty_ref is None or stocked is None or empty_ref.shape != stocked.shape:
        return []
    a = cv2.GaussianBlur(cv2.cvtColor(empty_ref, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(stocked, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(a, b)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = mask.shape
    min_area = min_area_frac * img_h * img_w
    found = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        px, py = int(bw * pad_frac), int(bh * pad_frac)
        nx1, ny1 = max(0, bx - px), max(0, by - py)
        nx2 = min(img_w, bx + bw + px)
        ny2 = min(img_h, by + bh + py + int(bh * pad_down_frac))
        found.append(((nx2 - nx1) * (ny2 - ny1),
                      {"x": nx1, "y": ny1, "w": nx2 - nx1, "h": ny2 - ny1}))
    found.sort(key=lambda t: -t[0])
    # left-to-right so auto-generated names track the physical shelf order
    return sorted((r for _, r in found[:max_boxes]), key=lambda r: r["x"])


def auto_tune_hsv(frame: np.ndarray, rect: dict, margin_frac: float = 0.35):
    """Fit an HSV band to the backdrop surrounding `rect` that deliberately
    does NOT match what is inside it. Returns (lower, upper, backdrop_match,
    object_match).

    Percentiles rather than mean±k*std: a textured floor or a shadow makes the
    standard deviation huge, which silently widens the band until it matches
    the product too and every slot reads 'empty'."""
    img_h, img_w = frame.shape[:2]
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    mx, my = max(8, int(w * margin_frac)), max(8, int(h * margin_frac))
    ox1, oy1 = max(0, x - mx), max(0, y - my)
    ox2, oy2 = min(img_w, x + w + mx), min(img_h, y + h + my)
    outer = frame[oy1:oy2, ox1:ox2]
    inside_bgr = frame[y:y + h, x:x + w]
    if outer.size == 0 or inside_bgr.size == 0:
        return None

    hsv_outer = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)
    ring_mask = np.ones(outer.shape[:2], dtype=bool)
    ring_mask[max(0, y - oy1):max(0, y - oy1) + h, max(0, x - ox1):max(0, x - ox1) + w] = False
    ring = hsv_outer[ring_mask]
    inside = cv2.cvtColor(inside_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if ring.size == 0:
        return None

    ignore_hue = ring[:, 1].mean() < 45  # grey/white backdrop: hue is just noise
    best = None
    for lo_p, hi_p in ((2, 98), (5, 95), (10, 90), (20, 80)):
        lower = np.clip(np.percentile(ring, lo_p, axis=0) - [3, 8, 10], 0, [180, 255, 255])
        upper = np.clip(np.percentile(ring, hi_p, axis=0) + [3, 8, 10], 0, [180, 255, 255])
        if ignore_hue:
            lower[0], upper[0] = 0, 180
        lower, upper = lower.astype(int).tolist(), upper.astype(int).tolist()
        lower, upper = _refine_band(ring, inside, lower, upper, free_hue=ignore_hue)
        backdrop_match, object_match = _frac_in_band(ring, lower, upper), _frac_in_band(inside, lower, upper)
        if best is None or (backdrop_match - object_match) > (best[2] - best[3]):
            best = (lower, upper, backdrop_match, object_match)
        if backdrop_match > 0.9 and object_match < 0.2:
            break
    return best


# ---------------------------------------------------------------- 1. camera
st.subheader("1. Shelf camera")
cameras = load_cameras()
default_url = cameras.get("shelf", {}).get("url", "")
url = st.text_input(
    "IP Webcam snapshot URL (e.g. http://192.168.1.23:8080/shot.jpg)",
    value="" if default_url == "simulate" else default_url,
)
c1, c2, c3 = st.columns(3)
if c1.button("Grab snapshot", type="primary"):
    try:
        frame = grab_frame(url)
        if frame is None:
            st.error("Got a response but couldn't decode it as an image.")
        else:
            st.session_state.frame = frame
            st.success(f"Got a {frame.shape[1]}x{frame.shape[0]} frame.")
    except Exception as exc:
        st.error(f"Can't reach {url} — {exc}")

if c2.button("Capture people-free background (8 shots)"):
    try:
        bg_est = BackgroundEstimator(max_frames=8)
        for _ in range(8):
            frame = grab_frame(url)
            if frame is not None:
                bg_est.add(frame)
            time.sleep(0.25)
        bg = bg_est.compute()
        if bg is None:
            st.error("Couldn't capture any snapshots.")
        else:
            st.session_state.frame = bg
            st.success("Background captured (in memory only).")
    except Exception as exc:
        st.error(f"Capture failed — {exc}")

if st.button("Reload saved calibration from disk"):
    n = _load_calibration_into_session()
    st.session_state.ui_version += 1
    st.success(f"Reloaded {n} slots "
               f"({len(st.session_state.slot_bands)} with their own band) from disk.")
    st.rerun()

if c3.button("Save URL to cameras.local.yaml"):
    local_path = CONFIG_DIR / "cameras.local.yaml"
    existing = load_yaml(local_path) if local_path.exists() else {}
    existing.setdefault("cameras", {})
    existing["cameras"]["shelf"] = {"role": "shelf", "url": url, "snapshot_interval_s": 3}
    save_yaml(local_path, existing)
    st.success("Saved — phase3_shelf.run will use this when run without --simulate.")

frame = st.session_state.frame

# ------------------------------------------------- 1b. auto-detect from empty/stocked
st.subheader("1b. Auto-detect slots (recommended)")
st.caption(
    "Show the camera the **empty** shelf once, then put the products back and let it "
    "find them. Each slot learns how its own background looks when empty, so slots on "
    "a wooden shelf work exactly like slots on the white wall. Don't move the phone "
    "between the two steps."
)
p1, p2 = st.columns(2)
pad_pct = p1.slider("Slot padding % (grow each box beyond the product)", 0, 60, 5)
pad_down_pct = p2.slider("Extra padding downward % (toward the shelf surface)", 0, 80, 10)
st.caption(
    "A slot is the patch of shelf a product is allowed to occupy, not the product's "
    "outline, so a little padding lets a slightly misplaced item still count as "
    "stocked. Too much and the box fills with backdrop while the product is still "
    "there — watch `stocked_reads` below and keep it under ~25%."
)

already = len(st.session_state.slots_list)
replace_ok = st.checkbox(
    f"Replace the {already} slot(s) I already have" if already else "Create slots from what is found",
    value=not already,
    help="Detection is a setup step. Once slots are calibrated they must stay put — a "
         "slot only means something if it is the same rectangle frame after frame. Run "
         "this on a shelf that is STOCKED; on an empty shelf it finds folds and shadows.",
)
if st.button("Detect objects in the CURRENT frame (no empty shelf needed)", type="primary",
             disabled=frame is None):
    boxes = detect_objects_here(frame, pad_frac=pad_pct / 100.0,
                                pad_down_frac=pad_down_pct / 100.0)
    if not boxes:
        st.error("No objects stood out from the backdrop. Is the shelf empty, or is the "
                 "backdrop as dark/colourful as the products?")
    elif already and not replace_ok:
        st.warning(
            f"Found {len(boxes)} object(s), but left your {already} calibrated slot(s) "
            "untouched. Tick the box above if you really want to redefine them."
        )
    else:
        conn_d = get_connection()
        db_slots = [row["slot"] for row in get_items(conn_d) if row["slot"]]
        id_by_slot = {row["slot"]: row["id"] for row in get_items(conn_d) if row["slot"]}
        new_slots, bands, report = [], {}, []
        for i, rect in enumerate(boxes):
            name = db_slots[i] if i < len(db_slots) else f"X{i + 1}"
            new_slots.append({"slot": name, "item_id": id_by_slot.get(name, 0), **rect})
            tuned = auto_tune_hsv(frame, rect)
            row = {"slot": name, "w": rect["w"], "h": rect["h"]}
            if tuned:
                lower, upper, backdrop_match, object_match = tuned
                bands[name] = (lower, upper)
                row |= {"empty_reads": f"{backdrop_match:.0%}",
                        "stocked_reads": f"{object_match:.0%}",
                        "usable": "yes" if backdrop_match > 0.8 and object_match < 0.3 else "marginal"}
            else:
                row |= {"empty_reads": "-", "stocked_reads": "-", "usable": "no band"}
            report.append(row)
        st.session_state.slots_list = new_slots
        st.session_state.slot_bands = bands
        st.session_state.ui_version += 1
        st.success(f"Detected {len(new_slots)} objects and fitted a backdrop band to each.")
        st.dataframe(pd.DataFrame(report), use_container_width=True)
        st.caption(
            "Each band is fitted from the backdrop ring *around* the box, so this needs a "
            "backdrop that differs from the product — which is exactly what the two-step "
            "flow below removes the need for. If a row says marginal, use Step A/Step B."
        )

st.markdown("**Two-step flow** (more reliable — each slot learns its own background):")
d1, d2 = st.columns(2)
if d1.button("Step A — capture EMPTY shelf reference"):
    try:
        ref = grab_frame(url)
        if ref is None:
            st.error("Got a response but couldn't decode it as an image.")
        else:
            st.session_state.empty_ref = ref
            st.session_state.frame = ref
            st.success("Empty-shelf reference captured (in memory only). "
                       "Now put the products back and hit Step B.")
    except Exception as exc:
        st.error(f"Can't reach {url} — {exc}")

if d2.button("Step B — put products back, then detect", type="primary",
             disabled=st.session_state.empty_ref is None):
    try:
        stocked = grab_frame(url)
    except Exception as exc:
        stocked = None
        st.error(f"Can't reach {url} — {exc}")
    if stocked is not None:
        boxes = detect_objects(st.session_state.empty_ref, stocked,
                               pad_frac=pad_pct / 100.0, pad_down_frac=pad_down_pct / 100.0)
        if not boxes:
            st.error(
                "Nothing changed between the two frames. Either the products weren't put "
                "back, or the camera re-exposed. Re-run Step A and try again."
            )
        else:
            st.session_state.frame = stocked
            db_slots = [row["slot"] for row in get_items(get_connection()) if row["slot"]]
            id_by_slot = {row["slot"]: row["id"] for row in get_items(get_connection()) if row["slot"]}
            new_slots, bands, report = [], {}, []
            for i, rect in enumerate(boxes):
                name = db_slots[i] if i < len(db_slots) else f"X{i + 1}"
                fitted = fit_band_from_empty(st.session_state.empty_ref, rect, stocked)
                if fitted is None:
                    continue
                lower, upper, empty_match, object_match = fitted
                bands[name] = (lower, upper)
                new_slots.append({"slot": name, "item_id": id_by_slot.get(name, 0), **rect})
                report.append({"slot": name, "empty_reads": f"{empty_match:.0%}",
                               "stocked_reads": f"{object_match:.0%}",
                               "usable": "yes" if empty_match > 0.75 and object_match < 0.4 else "marginal"})
            st.session_state.slots_list = new_slots
            st.session_state.slot_bands = bands
            st.session_state.ui_version += 1
            st.success(f"Detected {len(new_slots)} slots and fitted a backdrop band to each.")
            st.dataframe(pd.DataFrame(report), use_container_width=True)
            st.caption(
                "`empty_reads` is how much of the slot matches its band with the product "
                "gone — that is the fill% you'll see when it goes empty. `stocked_reads` is "
                "the same with the product present. You want high / low. Anything marked "
                "marginal won't flip reliably."
            )

if st.session_state.empty_ref is not None:
    st.caption("Empty-shelf reference is loaded in memory for this session.")

# ---------------------------------------------------------------- 2. click to define
st.subheader("2. Or click two corners to define one slot by hand")
st.caption(
    "Click the top-left corner of the object, then its bottom-right corner, then hit Apply. "
    "That fills the coordinates **and** auto-tunes the backdrop HSV from the wall around the box — "
    "no manual typing."
)

if frame is None:
    st.info("Grab a snapshot above first.")
else:
    click = streamlit_image_coordinates(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), key=f"picker_{st.session_state.ui_version}"
    )
    if click is not None:
        pt = (int(click["x"]), int(click["y"]))
        if pt != st.session_state.corner_last_click:
            st.session_state.corner_last_click = pt
            st.session_state.corner_pts = (st.session_state.corner_pts + [pt])[-2:]

    pts = st.session_state.corner_pts
    a1, a2, a3 = st.columns([2, 2, 1])
    slot_name = a1.text_input("Slot name for this box", value=st.session_state.slots_list[0]["slot"]
                              if st.session_state.slots_list else "A1")
    apply_clicked = a2.button("Apply box + auto-tune HSV", type="primary", disabled=len(pts) != 2)
    if a3.button("Clear clicks"):
        st.session_state.corner_pts, st.session_state.corner_last_click = [], None

    st.write(f"Points clicked: {pts}" if pts else "Points clicked: none yet")

    if apply_clicked and len(pts) == 2:
        (cx1, cy1), (cx2, cy2) = pts
        rect = {"x": min(cx1, cx2), "y": min(cy1, cy2),
                "w": abs(cx2 - cx1), "h": abs(cy2 - cy1)}
        name = slot_name.strip() or "A1"
        existing = next((s for s in st.session_state.slots_list if s["slot"] == name), None)
        if existing:
            existing.update(rect)
        else:
            st.session_state.slots_list.append({"slot": name, "item_id": 0, **rect})

        if st.session_state.empty_ref is not None:
            # An empty-shelf reference beats sampling the ring around the box.
            tuned = fit_band_from_empty(st.session_state.empty_ref, rect, frame)
            if tuned:
                st.session_state.slot_bands[name] = (tuned[0], tuned[1])
        else:
            tuned = auto_tune_hsv(frame, rect)
        if tuned:
            lower, upper, backdrop_match, object_match = tuned
            if st.session_state.empty_ref is None:
                st.session_state.hsv_lower, st.session_state.hsv_upper = lower, upper
            st.success(f"{name} set to x={rect['x']}, y={rect['y']}, w={rect['w']}, h={rect['h']} — "
                       f"HSV auto-tuned to {lower} … {upper}.")
            if backdrop_match > 0.8 and object_match < 0.3:
                st.info(f"Good separation: backdrop matches {backdrop_match:.0%}, "
                        f"object matches {object_match:.0%} — this box should read "
                        f"'ok' now and 'empty' once the object is removed.")
            else:
                st.warning(
                    f"Weak separation: backdrop matches {backdrop_match:.0%} but the object "
                    f"matches {object_match:.0%} too — the object's colour is too close to the "
                    "backdrop's, so 'empty' and 'ok' will be unreliable. Is the object actually "
                    "inside the box right now? If it is, use an object that contrasts more with "
                    "the surface behind it."
                )
        else:
            st.warning(f"{name} coordinates set, but HSV couldn't be sampled around that box.")
        st.session_state.corner_pts, st.session_state.corner_last_click = [], None
        st.session_state.ui_version += 1

# ---------------------------------------------------------------- 3. slots table
st.subheader("3. Slots (edit by hand if needed)")
edited = st.data_editor(
    pd.DataFrame(st.session_state.slots_list, columns=SLOT_COLUMNS),
    num_rows="dynamic", use_container_width=True,
    key=f"slot_editor_{st.session_state.ui_version}",
)
current_slots = [
    {"slot": str(r["slot"]), "item_id": int(r["item_id"]), "x": int(r["x"]), "y": int(r["y"]),
     "w": int(r["w"]), "h": int(r["h"])}
    for _, r in edited.dropna().iterrows() if str(r["slot"]).strip()
]
st.session_state.slots_list = current_slots
# Re-attach each slot's own band (kept out of the editable table).
for s in current_slots:
    band = st.session_state.slot_bands.get(s["slot"])
    if band:
        s["hsv_lower"], s["hsv_upper"] = band

conn = get_connection()
missing_in_db = {s["slot"] for s in current_slots} - {row["slot"] for row in get_items(conn) if row["slot"]}
if missing_in_db:
    st.warning(f"No stock-DB item maps to these slots, so no alerts will fire for them: "
               f"{', '.join(sorted(missing_in_db))}")

# ---------------------------------------------------------------- 4. HSV + thresholds
st.subheader("4. Backdrop HSV and state thresholds")
st.caption(
    "Auto-filled by step 2. Re-tune if the lighting changes — a range fitted to one "
    "exposure stops matching after the camera re-exposes."
)
v = st.session_state.ui_version
hc1, hc2, hc3 = st.columns(3)
h_lower = [
    hc1.number_input("H lower", 0, 180, st.session_state.hsv_lower[0], key=f"hl{v}"),
    hc2.number_input("S lower", 0, 255, st.session_state.hsv_lower[1], key=f"sl{v}"),
    hc3.number_input("V lower", 0, 255, st.session_state.hsv_lower[2], key=f"vl{v}"),
]
h_upper = [
    hc1.number_input("H upper", 0, 180, st.session_state.hsv_upper[0], key=f"hu{v}"),
    hc2.number_input("S upper", 0, 255, st.session_state.hsv_upper[1], key=f"su{v}"),
    hc3.number_input("V upper", 0, 255, st.session_state.hsv_upper[2], key=f"vu{v}"),
]
st.session_state.hsv_lower, st.session_state.hsv_upper = h_lower, h_upper

if st.button("Re-tune HSV from current frame") and frame is not None and current_slots:
    tuned = auto_tune_hsv(frame, current_slots[0])
    if tuned:
        st.session_state.hsv_lower, st.session_state.hsv_upper = tuned[0], tuned[1]
        st.session_state.ui_version += 1
        st.rerun()

settings = load_settings()
shelf_settings = settings.get("shelf", {})
t1, t2, t3 = st.columns(3)
empty_pct = t1.number_input("Empty threshold %", 0, 100, int(shelf_settings.get("empty_threshold_pct", 60)))
low_pct = t2.number_input("Low threshold %", 0, 100, int(shelf_settings.get("low_threshold_pct", 30)))
confirm_frames = t3.number_input("Confirm frames", 1, 10, int(shelf_settings.get("confirm_frames", 2)))

# ---------------------------------------------------------------- 5. preview
st.subheader("5. Preview")
st.caption("Move the object out of the box, hit Grab snapshot above, and this should flip to `empty`.")
if frame is None:
    st.info("Grab a snapshot above to preview.")
elif not current_slots:
    st.info("Define a slot above to preview.")
else:
    preview = frame.copy()
    detector = GapDetector(current_slots, h_lower, h_upper,
                           empty_threshold_pct=empty_pct, low_threshold_pct=low_pct,
                           confirm_frames=1)
    rows = []
    for slot in current_slots:
        fill_pct = detector._backdrop_fraction(frame, slot)
        state = "empty" if fill_pct > empty_pct else ("low" if fill_pct > low_pct else "ok")
        color = {"empty": (0, 0, 255), "low": (0, 200, 255), "ok": (0, 180, 0)}[state]
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 3)
        label = f"{slot['slot']} {fill_pct:.0f}% {state}"
        pos = (x, max(16, y - 8))
        cv2.putText(preview, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        rows.append({"slot": slot["slot"], "backdrop_fill_pct": round(fill_pct, 1),
                     "predicted_state": state,
                     "band": "own" if slot["slot"] in st.session_state.slot_bands else "global"})
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------------------------------------------------------- 6. save
st.subheader("6. Save calibration")
if st.button("Save slots + HSV + thresholds", type="primary"):
    if not current_slots:
        st.error("Define at least one slot first.")
    else:
        save_slots({
            "backdrop_hsv_lower": [int(x) for x in h_lower],
            "backdrop_hsv_upper": [int(x) for x in h_upper],
            "slots": [
                {**s,
                 **({"hsv_lower": [int(v) for v in st.session_state.slot_bands[s["slot"]][0]],
                     "hsv_upper": [int(v) for v in st.session_state.slot_bands[s["slot"]][1]]}
                    if s["slot"] in st.session_state.slot_bands else {})}
                for s in current_slots
            ],
        })
        settings["shelf"] = {"empty_threshold_pct": int(empty_pct),
                             "low_threshold_pct": int(low_pct),
                             "confirm_frames": int(confirm_frames)}
        save_yaml(CONFIG_DIR / "settings.yaml", settings)
        st.success("Saved. Restart phase3_shelf.run to pick up the new calibration.")
