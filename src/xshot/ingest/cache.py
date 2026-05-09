"""Shared HTTP-friendly fetch helpers and simple rate limiting."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def stable_cache_key(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"


def throttle(last_call: list[float], min_interval_s: float) -> None:
    """Mutable list[float] keeps monotonic last call time for rate limiting."""
    now = time.monotonic()
    wait = min_interval_s - (now - last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_call[0] = time.monotonic()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
