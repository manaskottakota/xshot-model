"""Append-only run history + comparison against the previous logged run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_RUN_KEYS = frozenset({"roc_auc", "log_loss", "accuracy", "f1"})


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_last_completed_run(history_path: Path) -> dict[str, Any] | None:
    if not history_path.exists():
        return None
    lines = history_path.read_text().splitlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def format_previous_run_compare(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    label: str = "calibrated_primary_test",
) -> str:
    if previous is None:
        return "(no prior run in metrics history — deltas unavailable)"
    prev_row = previous.get(label)
    cur_row = current.get(label)
    if not isinstance(prev_row, Mapping) or not isinstance(cur_row, Mapping):
        return "(previous/current run missing calibrated_primary_test slice)"
    parts = ["vs_previous_run:", f"  prior_timestamp: {previous.get('ts')!r}", f"  prior_out_dir: {previous.get('out_dir')!r}"]
    for k in sorted(_RUN_KEYS & cur_row.keys() & prev_row.keys()):
        a = float(cur_row[k])
        b = float(prev_row[k])
        d = a - b
        parts.append(f"  Δ{k}: {d:+.5f}  (was {b:.5f}, now {a:.5f})")
    return "\n".join(parts)


def append_training_run_record(
    history_path: Path,
    *,
    out_dir: Path,
    manifest: Mapping[str, Any],
    calibrated_test_row: Mapping[str, Any],
    primary_model: str,
    omit_features: list[str] | None,
    ablation_variant: str | None,
    feature_diagnostics_path: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    rec: dict[str, Any] = {
        "ts": _utc_iso(),
        "out_dir": str(out_dir),
        "manifest_features": manifest.get("features"),
        "omit_features": sorted(omit_features or []),
        "ablation_variant": ablation_variant or "",
        "primary_model": primary_model,
        "calibrated_primary_test": {k: calibrated_test_row[k] for k in _RUN_KEYS if k in calibrated_test_row},
        "manifest": dict(manifest),
    }
    if feature_diagnostics_path:
        rec["feature_diagnostics_path"] = feature_diagnostics_path
    if extra:
        rec.update(extra)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
