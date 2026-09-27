#!/usr/bin/env python3
"""Seeds ~10 Indian grocery items and 30 days of simulated sales history.

The sales history is entirely synthetic (weekday and month-start demand
effects layered on a base rate) — labelled as such in the dashboard.
"""
from __future__ import annotations

import random
import time

from storesmart.stock.db import DEFAULT_DB_PATH, get_connection

ITEMS = [
    # name, category, slot, shelf_qty, store_qty, reorder_level, supplier, visit_weekday(0=Mon), lead_days, price, base_daily_demand
    ("Atta (5kg)", "Grocery", "A1", 12, 20, 8, "Shree Traders", 0, 1, 220, 4),
    ("Rice (5kg)", "Grocery", "A2", 10, 15, 6, "Shree Traders", 0, 1, 260, 3),
    ("Toor Dal (1kg)", "Grocery", "A3", 15, 25, 8, "Shree Traders", 0, 1, 140, 5),
    ("Sunflower Oil (1L)", "Grocery", "B1", 10, 12, 6, "Ganesh Distributors", 2, 2, 165, 4),
    ("Sugar (1kg)", "Grocery", "B2", 14, 18, 8, "Ganesh Distributors", 2, 2, 48, 6),
    ("Tea (250g)", "Beverages", "C1", 12, 16, 6, "Ganesh Distributors", 2, 2, 130, 4),
    ("Biscuits (Parle-G)", "Snacks", "C2", 20, 30, 10, "Krishna Snacks Co", 4, 1, 10, 12),
    ("Namkeen (200g)", "Snacks", "C3", 15, 20, 8, "Krishna Snacks Co", 4, 1, 45, 7),
    ("Soap (Lifebuoy)", "Personal Care", "D1", 18, 24, 8, "Ganesh Distributors", 2, 2, 35, 5),
    ("Milk Packet (500ml)", "Dairy", "D2", 20, 10, 12, "Local Dairy", 6, 0, 28, 15),
]

WEEKDAY_FACTOR = [1.0, 0.9, 0.9, 1.0, 1.1, 1.4, 1.3]  # Mon..Sun, weekend bump


def seed(db_path=DEFAULT_DB_PATH, days: int = 30, seed_value: int = 42) -> None:
    random.seed(seed_value)
    conn = get_connection(db_path)
    conn.execute("DELETE FROM sales")
    conn.execute("DELETE FROM stock_moves")
    conn.execute("DELETE FROM items")
    conn.commit()

    item_ids = {}
    for name, category, slot, shelf_qty, store_qty, reorder_level, supplier, weekday, lead_days, price, _ in ITEMS:
        cur = conn.execute(
            "INSERT INTO items(name, category, slot, shelf_qty, store_qty, reorder_level, "
            "supplier, supplier_visit_weekday, lead_days, price) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (name, category, slot, shelf_qty, store_qty, reorder_level, supplier, weekday, lead_days, price),
        )
        item_ids[name] = cur.lastrowid
    conn.commit()

    now = time.time()
    day_s = 86400
    for day_offset in range(days, 0, -1):
        day_ts = now - day_offset * day_s
        weekday = time.gmtime(day_ts).tm_wday
        is_month_start = time.gmtime(day_ts).tm_mday <= 3
        for name, *_rest, base_demand in ITEMS:
            factor = WEEKDAY_FACTOR[weekday] * (1.25 if is_month_start else 1.0)
            n_sales = max(0, round(random.gauss(base_demand * factor, base_demand * 0.25)))
            for _ in range(n_sales):
                ts = day_ts + random.uniform(0, day_s)
                qty = random.choice([1, 1, 1, 2])
                conn.execute(
                    "INSERT INTO sales(item_id, qty, ts) VALUES (?, ?, ?)",
                    (item_ids[name], qty, ts),
                )
    conn.commit()
    conn.close()
    print(f"Seeded {len(ITEMS)} items and {days} days of simulated sales into {db_path}")


if __name__ == "__main__":
    seed()
