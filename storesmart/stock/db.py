"""Stock/billing SQLite schema, sharing the same database file as the event
bus but its own tables. WAL mode, one connection helper reused everywhere."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path("data/storesmart.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    slot TEXT,
    shelf_qty INTEGER NOT NULL DEFAULT 0,
    store_qty INTEGER NOT NULL DEFAULT 0,
    reorder_level INTEGER NOT NULL DEFAULT 5,
    supplier TEXT NOT NULL,
    supplier_visit_weekday INTEGER NOT NULL, -- 0=Monday .. 6=Sunday
    lead_days INTEGER NOT NULL DEFAULT 1,
    price REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id),
    qty INTEGER NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sales_item_ts ON sales(item_id, ts);

CREATE TABLE IF NOT EXISTS stock_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES items(id),
    kind TEXT NOT NULL,   -- 'refill' (storeroom->shelf), 'delivery' (supplier->storeroom), 'sale'
    qty INTEGER NOT NULL,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alert_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    item_name TEXT,
    ts REAL NOT NULL
);
"""


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def record_sale(conn: sqlite3.Connection, item_id: int, qty: int = 1) -> None:
    now = time.time()
    conn.execute("INSERT INTO sales(item_id, qty, ts) VALUES (?, ?, ?)", (item_id, qty, now))
    conn.execute("INSERT INTO stock_moves(item_id, kind, qty, ts) VALUES (?, 'sale', ?, ?)",
                 (item_id, qty, now))
    conn.execute("UPDATE items SET shelf_qty = MAX(0, shelf_qty - ?) WHERE id = ?", (qty, item_id))
    conn.commit()


def refill_from_storeroom(conn: sqlite3.Connection, item_id: int, qty: int) -> None:
    now = time.time()
    row = conn.execute("SELECT store_qty FROM items WHERE id=?", (item_id,)).fetchone()
    qty = min(qty, row["store_qty"]) if row else 0
    if qty <= 0:
        return
    conn.execute("UPDATE items SET shelf_qty = shelf_qty + ?, store_qty = store_qty - ? WHERE id=?",
                 (qty, qty, item_id))
    conn.execute("INSERT INTO stock_moves(item_id, kind, qty, ts) VALUES (?, 'refill', ?, ?)",
                 (item_id, qty, now))
    conn.commit()


def record_delivery(conn: sqlite3.Connection, item_id: int, qty: int) -> None:
    now = time.time()
    conn.execute("UPDATE items SET store_qty = store_qty + ? WHERE id=?", (qty, item_id))
    conn.execute("INSERT INTO stock_moves(item_id, kind, qty, ts) VALUES (?, 'delivery', ?, ?)",
                 (item_id, qty, now))
    conn.commit()


def get_items(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM items ORDER BY category, name").fetchall()


def get_item(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()


def get_item_by_slot(conn: sqlite3.Connection, slot: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM items WHERE slot=?", (slot,)).fetchone()


def recent_sales(conn: sqlite3.Connection, item_id: int, since_ts: float) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sales WHERE item_id=? AND ts >= ? ORDER BY ts", (item_id, since_ts)
    ).fetchall()


def was_alerted_recently(conn: sqlite3.Connection, kind: str, item_name: str | None, window_min: float) -> bool:
    cutoff = time.time() - window_min * 60
    row = conn.execute(
        "SELECT 1 FROM alert_log WHERE kind=? AND (item_name=? OR (item_name IS NULL AND ? IS NULL)) "
        "AND ts >= ? LIMIT 1",
        (kind, item_name, item_name, cutoff),
    ).fetchone()
    return row is not None


def log_alert(conn: sqlite3.Connection, kind: str, item_name: str | None) -> None:
    conn.execute("INSERT INTO alert_log(kind, item_name, ts) VALUES (?, ?, ?)",
                 (kind, item_name, time.time()))
    conn.commit()
