"""Shelf slot definitions: rectangles on the shelf snapshot, each mapped to
an item ID in the stock DB. Loaded from / saved to config/shelf_slots.json."""
from __future__ import annotations

from storesmart.common.config import load_json, save_json

DEFAULT_PATH = "config/shelf_slots.json"
EXAMPLE_PATH = "config/shelf_slots.example.json"


def load_slots(path: str = DEFAULT_PATH) -> dict:
    try:
        return load_json(path)
    except FileNotFoundError:
        return load_json(EXAMPLE_PATH)


def save_slots(data: dict, path: str = DEFAULT_PATH) -> None:
    save_json(path, data)
