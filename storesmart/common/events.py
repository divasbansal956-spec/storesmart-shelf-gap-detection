"""Event schema + gate.

Every event that leaves a perception module must pass through `validate_event`
before it is written to the bus. This is the privacy boundary: only small,
whitelisted JSON events may persist. Anything else (images, embeddings,
unexpected fields, oversized payloads) is rejected and counted.
"""
from __future__ import annotations

import json
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

MAX_EVENT_BYTES = 1024


class RejectedEvent(Exception):
    def __init__(self, reason: str, raw: object):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t: str
    cam: Optional[str] = None


class EntryEvent(_Base):
    type: Literal["entry"]


class ExitEvent(_Base):
    type: Literal["exit"]


class QueueEvent(_Base):
    type: Literal["queue"]
    length: int
    serving: int
    counters: int
    wait_s: float
    forecast_wait_s: float


class ServedEvent(_Base):
    type: Literal["served"]
    service_s: float


class ShelfStatusEvent(_Base):
    type: Literal["shelf_status"]
    slot: str
    fill_pct: float
    state: Literal["ok", "low", "empty"]


class PositionEvent(_Base):
    type: Literal["position"]
    x_ft: float
    y_ft: float
    track: int


class DwellEvent(_Base):
    type: Literal["dwell"]
    zone: str
    dwell_s: float


class AlertEvent(_Base):
    type: Literal["alert"]
    kind: Literal["open_counter", "refill", "reorder", "mismatch", "layout"]
    msg: str
    item: Optional[str] = None
    severity: Literal["info", "warn", "urgent"] = "info"


class SummaryEvent(_Base):
    type: Literal["summary"]
    in_: int = Field(alias="in")
    out: int
    inside: int
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


EVENT_TYPES = {
    "entry": EntryEvent,
    "exit": ExitEvent,
    "queue": QueueEvent,
    "served": ServedEvent,
    "shelf_status": ShelfStatusEvent,
    "position": PositionEvent,
    "dwell": DwellEvent,
    "alert": AlertEvent,
    "summary": SummaryEvent,
}

AnyEvent = Union[
    EntryEvent, ExitEvent, QueueEvent, ServedEvent, ShelfStatusEvent,
    PositionEvent, DwellEvent, AlertEvent, SummaryEvent,
]

# Fields that must never appear on any event, regardless of type — a
# belt-and-braces check against image/blob smuggling via extra fields.
_BANNED_SUBSTRINGS = ("image", "frame", "jpg", "jpeg", "png", "base64", "blob", "embedding")


class EventGate:
    """Validates raw dicts against the schema. Tracks accept/reject counts."""

    def __init__(self):
        self.accepted = 0
        self.rejected = 0

    def validate(self, raw: dict) -> AnyEvent:
        try:
            self._check_size(raw)
            self._check_banned_fields(raw)
            etype = raw.get("type")
            model = EVENT_TYPES.get(etype)
            if model is None:
                raise RejectedEvent(f"unknown event type: {etype!r}", raw)
            event = model.model_validate(raw)
        except RejectedEvent:
            self.rejected += 1
            raise
        except Exception as exc:  # pydantic ValidationError, etc.
            self.rejected += 1
            raise RejectedEvent(str(exc), raw) from exc
        self.accepted += 1
        return event

    @staticmethod
    def _check_size(raw: dict) -> None:
        size = len(json.dumps(raw, separators=(",", ":")).encode("utf-8"))
        if size > MAX_EVENT_BYTES:
            raise RejectedEvent(f"event too large: {size} bytes > {MAX_EVENT_BYTES}", raw)

    @staticmethod
    def _check_banned_fields(raw: dict) -> None:
        for key, value in raw.items():
            lk = key.lower()
            if any(b in lk for b in _BANNED_SUBSTRINGS):
                raise RejectedEvent(f"banned field: {key!r}", raw)
            if isinstance(value, str) and len(value) > 200:
                raise RejectedEvent(f"suspiciously long string field: {key!r}", raw)
