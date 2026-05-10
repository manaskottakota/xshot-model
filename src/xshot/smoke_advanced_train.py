"""Tiny core+advanced fit for CI / local smoke — proves advanced columns move ``predict_proba``."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xshot.models.trainers import build_models, fit_classifier_pipeline
from xshot.viz.inference_row import inference_X_from_payload


def _row_payload(i: int, defender_ft: float) -> dict:
    return {
        "loc_x_ft": -15.2 + (i % 9) * 0.11,
        "loc_y_ft": 17.4 + (i % 6) * 0.09,
        "is_three": True,
        "shot_archetype": "pullup",
        "score_diff": -3 + (i % 5),
        "period": 4,
        "minutes_remaining": 5.0,
        "seconds_remaining": float(20 + (i % 40)),
        "shot_clock_known": True,
        "shot_clock_seconds": 8.0,
        "player_profile": "league_average",
        "defender_distance_ft": defender_ft,
        "defender_contest_azimuth_deg": float((i % 25) - 12),
        "dribbles_before_shot": 0.6 + (i % 7) * 0.15,
        "touch_time_sec": 1.1 + (i % 5) * 0.12,
        "time_since_catch": 0.7 + (i % 4) * 0.1,
        "distance_traveled_before_shot": 1.0 + (i % 8) * 0.2,
        "rest_days_since_prev_game": 1.5 + (i % 4) * 0.5,
        "is_back_to_back": bool(i % 11 == 0),
    }


def fit_smoke_advanced_classifier(
    out_path: Path,
    *,
    n_rows: int = 420,
    val_frac: float = 0.18,
    seed: int = 42,
) -> Path:
    """
    Fit ``logistic_regression`` on synthetic rows whose labels correlate with defender distance.

    Not a quality model — only demonstrates that the advanced feature schema is wired into
    a sklearn ``Pipeline`` the same way as production training.
    """
    rng = np.random.RandomState(seed)
    xs: list[pd.DataFrame] = []
    ys: list[int] = []
    for i in range(n_rows):
        d = 1.15 + (i / max(n_rows - 1, 1)) * 11.5
        # Strong monotone rule + light label noise so logistic picks up ``defender_distance_ft``.
        base_y = 1 if d > 5.8 else 0
        y = base_y if rng.rand() > 0.12 else 1 - base_y
        xs.append(inference_X_from_payload(_row_payload(i, d), features="core+advanced"))
        ys.append(y)

    X = pd.concat(xs, ignore_index=True)
    y = np.asarray(ys, dtype=np.int64)
    n_val = max(int(len(X) * val_frac), 24)
    ix = rng.permutation(len(X))
    tr_ix, va_ix = ix[n_val:], ix[:n_val]

    models = build_models(X.iloc[tr_ix], ["shot_zone_basic"])
    name = "logistic_regression"
    pipe = models[name]
    fit_classifier_pipeline(
        name,
        pipe,
        X.iloc[tr_ix],
        y[tr_ix],
        X.iloc[va_ix],
        y[va_ix],
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return out_path
