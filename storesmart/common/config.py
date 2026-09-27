"""Config loading helpers for YAML/JSON config files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_json(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_yaml(path: Path | str, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def save_json(path: Path | str, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_cameras(path: Path | str | None = None) -> dict[str, dict]:
    """Load config/cameras.yaml, preferring a gitignored cameras.local.yaml
    with real phone IPs if present."""
    local = CONFIG_DIR / "cameras.local.yaml"
    if path is None and local.exists():
        path = local
    elif path is None:
        path = CONFIG_DIR / "cameras.yaml"
    cfg = load_yaml(path)
    return cfg.get("cameras", {})


def load_settings(path: Path | str | None = None) -> dict:
    path = path or (CONFIG_DIR / "settings.yaml")
    return load_yaml(path)


def save_cameras_local(cameras: dict[str, dict]) -> Path:
    """Write camera sources to config/cameras.local.yaml, which is gitignored
    so real phone IPs never reach the repository."""
    path = CONFIG_DIR / "cameras.local.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Local camera sources — gitignored, never commit real IPs.\n"
        "# Written by the dashboard's Control page; edit by hand if you prefer.\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump({"cameras": cameras}, fh, sort_keys=False)
    return path
