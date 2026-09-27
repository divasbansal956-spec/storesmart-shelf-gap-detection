"""Rule-based alerts combining shelf state (from the gap detector) with the
stock DB: refill from storeroom, reorder from distributor, and stock
mismatch. Alerts are deduplicated so the same one isn't repeated within
`dedupe_window_min` minutes.
"""
from __future__ import annotations

import datetime
import time

from storesmart.common.bus import EventBus
from storesmart.stock.db import get_item_by_slot, log_alert, recent_sales, was_alerted_recently
from storesmart.stock.forecast import forecast_run_out

MISMATCH_RECENT_SALES_WINDOW_S = 3600  # no sale in the last hour + shelf shows empty -> suspicious


def evaluate_shelf_status(conn, bus: EventBus, slot: str, state: str, dedupe_window_min: float = 15) -> None:
    """Called whenever the gap detector reports a confirmed slot state
    change. Emits refill/reorder/mismatch alerts as appropriate."""
    item = get_item_by_slot(conn, slot)
    if item is None:
        return
    item = dict(item)

    if state in ("empty", "low"):
        _maybe_refill_alert(conn, bus, item, dedupe_window_min)
        _maybe_mismatch_alert(conn, bus, item, state, dedupe_window_min)

    _maybe_reorder_alert(conn, bus, item, dedupe_window_min)


def _maybe_refill_alert(conn, bus: EventBus, item: dict, dedupe_window_min: float) -> None:
    if item["store_qty"] <= 0:
        return
    if was_alerted_recently(conn, "refill", item["name"], dedupe_window_min):
        return
    bus.emit({
        "type": "alert", "kind": "refill", "severity": "warn", "item": item["name"],
        "msg": f"Refill {item['name']} from storeroom ({item['store_qty']} available).",
    })
    log_alert(conn, "refill", item["name"])


def _maybe_reorder_alert(conn, bus: EventBus, item: dict, dedupe_window_min: float) -> None:
    forecast = forecast_run_out(conn, item)
    below_reorder = (item["shelf_qty"] + item["store_qty"]) <= item["reorder_level"]
    if not (forecast.will_run_out_before_visit or below_reorder):
        return
    if was_alerted_recently(conn, "reorder", item["name"], dedupe_window_min):
        return
    visit_str = forecast.next_supplier_visit.strftime("%A %d %b")
    order_by_str = forecast.order_by.strftime("%A %d %b")
    bus.emit({
        "type": "alert", "kind": "reorder", "severity": "urgent" if below_reorder else "warn",
        "item": item["name"],
        "msg": f"Order {item['name']} by {order_by_str} — {item['supplier']} visits {visit_str}.",
    })
    log_alert(conn, "reorder", item["name"])


def _maybe_mismatch_alert(conn, bus: EventBus, item: dict, state: str, dedupe_window_min: float) -> None:
    if state != "empty":
        return
    since = time.time() - MISMATCH_RECENT_SALES_WINDOW_S
    sold_recently = len(recent_sales(conn, item["id"], since)) > 0
    if sold_recently or item["shelf_qty"] <= 0:
        return  # empty shelf is explained by sales, or the DB agrees it's empty
    if was_alerted_recently(conn, "mismatch", item["name"], dedupe_window_min):
        return
    bus.emit({
        "type": "alert", "kind": "mismatch", "severity": "warn", "item": item["name"],
        "msg": f"Stock mismatch: check {item['name']} — shelf looks empty but "
               f"{item['shelf_qty']} units are recorded on the shelf with no recent sales.",
    })
    log_alert(conn, "mismatch", item["name"])
