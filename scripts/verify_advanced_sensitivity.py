#!/usr/bin/env python3
"""Compare core+advanced predictions when defender proximity sweeps materially wide.

Exits 3 if the artifact was trained with ``features=core`` (advanced columns are dropped
at ``ColumnTransformer`` inference — sweeps look “dead”).

Exits 2 if calibrated probabilities barely move for a core+advanced model.

Usage::

    python scripts/verify_advanced_sensitivity.py
    python scripts/verify_advanced_sensitivity.py --smoke-train
    python scripts/verify_advanced_sensitivity.py --model-path artifacts/run_default/xgboost.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from xshot.smoke_advanced_train import fit_smoke_advanced_classifier
from xshot.viz.inference_row import inference_X_from_payload, resolve_model_path

BASE_PAYLOAD = {
    "loc_x_ft": -16.82,
    "loc_y_ft": 18.25,
    "is_three": True,
    "shot_archetype": "pullup",
    "score_diff": -4,
    "period": 4,
    "minutes_remaining": 4.75,
    "seconds_remaining": 18.25,
    "shot_clock_known": True,
    "shot_clock_seconds": 7.2,
    "player_profile": "league_average",
    "defender_contest_azimuth_deg": -8.0,
    "dribbles_before_shot": 1.42,
    "touch_time_sec": 1.75,
}


def _manifest_features(model_path: Path) -> str | None:
    metrics = model_path.parent / "metrics.json"
    if not metrics.exists():
        return None
    try:
        blob = json.loads(metrics.read_text())
    except json.JSONDecodeError:
        return None
    man = blob.get("manifest") or {}
    return man.get("features")


def _pipe_accepts_advanced(clf) -> bool:
    """Best-effort: infer whether fitted numeric block includes advanced columns."""
    try:
        pre = clf.named_steps.get("pre")
        if pre is None:
            return False
        names = pre.get_feature_names_out()
        blob = "".join(names)
        return "defender_distance_ft" in blob
    except Exception:
        return False


def sweep_probs(clf) -> tuple[float, float, float]:
    tight = inference_X_from_payload(
        {**BASE_PAYLOAD, "defender_distance_ft": 1.15},
        features="core+advanced",
    )
    open_ = inference_X_from_payload(
        {**BASE_PAYLOAD, "defender_distance_ft": 11.92},
        features="core+advanced",
    )
    drib = inference_X_from_payload(
        {
            **BASE_PAYLOAD,
            "defender_distance_ft": 6.82,
            "dribbles_before_shot": 0.72,
            "touch_time_sec": 1.92,
            "distance_traveled_before_shot": 11.92,
            "time_since_catch": 3.92,
            "rest_days_since_prev_game": 0.92,
            "is_back_to_back": True,
        },
        features="core+advanced",
    )
    p_t = clf.predict_proba(tight)[0, 1]
    p_o = clf.predict_proba(open_)[0, 1]
    p_d = clf.predict_proba(drib)[0, 1]
    return float(p_t), float(p_o), float(p_d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument(
        "--def-delta-floor",
        type=float,
        default=5.0e-3,
        help="Minimum |Δp| tight vs open defender distance.",
    )
    ap.add_argument(
        "--smoke-train",
        action="store_true",
        help="Fit a tiny logistic on synthetic core+advanced rows, then verify sensitivity.",
    )
    ap.add_argument(
        "--smoke-out",
        type=Path,
        default=Path("artifacts/smoke_advanced/logistic_regression.joblib"),
    )
    args = ap.parse_args()

    if args.smoke_train:
        path = fit_smoke_advanced_classifier(args.smoke_out)
        print(f"[smoke-train] wrote {path}")
        args.model_path = str(path)

    explicit = Path(args.model_path).resolve() if args.model_path else None
    model_path = resolve_model_path(explicit)
    if model_path is None:
        print(
            "[verify_advanced_sensitivity] No model artifact — use --smoke-train or train "
            "`xshot-train --features core+advanced`."
        )
        return 0

    clf = joblib.load(model_path)
    mf = _manifest_features(model_path)
    if mf == "core" and not args.smoke_train:
        print(
            f"artifact={model_path.name}\n"
            f"[FATAL] metrics.json reports features={mf!r}. Advanced columns are omitted at "
            f"inference — retrain with --features core+advanced or pass --smoke-train.\n"
            f"        (exit 3)"
        )
        return 3

    if mf != "core+advanced" and not _pipe_accepts_advanced(clf) and not args.smoke_train:
        print(
            f"artifact={model_path.name}\n"
            f"[WARN] Could not confirm advanced features in preprocessor "
            f"(metrics features={mf!r}). Continuing — if this fails, retrain core+advanced.\n"
        )

    p_t, p_o, p_d = sweep_probs(clf)
    defender_swing = abs(p_o - p_t)
    drib_swing_vs_mid = abs(p_d - np.median(np.asarray([p_t, p_o], dtype=float)))
    delta_pp = defender_swing * 100

    print(f"artifact={model_path.name}")
    print(f"Tight FG p={p_t:.5f}, open-ish FG p={p_o:.5f}, dribble-shift FG p={p_d:.5f}")
    print(f"Δp(defender sweep) raw={defender_swing:.5f}  ({delta_pp:.3f} pp)")
    print(f"Movement delta vs midpoint={drib_swing_vs_mid:.5f}")

    if defender_swing >= args.def_delta_floor:
        return 0

    print("\n[HINT] Failure modes:")
    print("- Trained with core only (see exit 3 path above).")
    print("- Over-regularized / miscalibrated artifact — try --smoke-train to validate plumbing.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
