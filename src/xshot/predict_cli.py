"""Synthetic “one shot” probability helper for demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xshot.datasets import PRIOR_LAST_N_FG_COL, feature_columns


def _example_row(features: str) -> pd.DataFrame:
    """Hand-built row aligned with ``feature_columns``."""
    num, cat = feature_columns(features)
    base = {c: 0.0 for c in num}
    base.update(
        {
            "loc_x_ft": -22.0,
            "loc_y_ft": 8.0,
            "shot_angle_rad": -1.15,
            "shot_distance_ft": 22.0,
            "is_restricted_area": 0,
            "is_corner_three": 1,
            "is_midrange": 0,
            "period": 4,
            "secs_left_in_quarter": 90.0,
            "early_in_quarter": 0,
            "shooting_team_is_home": 1.0,
            "score_diff_shooting_perspective_safe": -2.0,
            "shooting_team_ahead": 0,
            "shooting_team_trailing": 1,
            "clutch_time": 1,
            "is_playoffs": 0,
            "shot_clock_remaining": -1.0,
            "shot_clock_known": 0,
            "shot_style_layup": 0,
            "shot_style_dunk": 0,
            "shot_style_hook": 0,
            "shot_style_fadeaway": 0,
            "shot_style_jumper": 1,
            "shot_style_pullup": 0,
            "shot_style_stepback": 0,
            "is_three": 1,
            "prior_cum_fg_pct": 0.42,
            "prior_three_attempt_share": 0.55,
            "prior_attempts_global": 400.0,
            PRIOR_LAST_N_FG_COL: 0.41,
        }
    )
    if features == "core+advanced":
        base.update(
            {
                "defender_distance_ft": 3.5,
                "def_contest_open_bucket": 1.0,
                "defender_rel_angle_rad": 0.2,
                "defender_geom_known": 0,
                "dribbles_before_shot": 0.0,
                "touch_time_sec": 1.2,
                "elapsed_game_sec_approx": 2400.0,
                "player_load_game_min_approx": 40.0,
                "rest_days_since_prev_game": 2.0,
                "is_back_to_back": 0,
                "tracking_merge_ok": 1,
            }
        )
    row = pd.DataFrame([base])[num]
    zones = pd.DataFrame({"shot_zone_basic": ["Left Corner 3"]})[cat]
    return pd.concat([row, zones], axis=1)[num + cat]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="xshot-predict-one")
    p.add_argument(
        "--features",
        choices=["core", "core+advanced"],
        default="core",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="``joblib`` dump of fitted ``Pipeline``. Defaults to histogram/xgboost artifact if present.",
    )
    args = p.parse_args(argv)

    model_path = args.model_path
    if model_path is None:
        for cand in (
            Path("artifacts/run_default/xshot_primary_calibrated.joblib"),
            Path("artifacts/run_default/xshot_primary.joblib"),
            Path("artifacts/run_default/histogram_gradient_boosting.joblib"),
            Path("artifacts/run_default/xgboost.joblib"),
        ):
            if cand.exists():
                model_path = cand
                break
    if model_path is None or not model_path.exists():
        raise SystemExit(
            "No model artifact — train first (`python scripts/train.py`) or pass `--model-path`."
        )

    clf = joblib.load(model_path)
    shot = _example_row(args.features)
    prob = clf.predict_proba(shot)[0, 1]
    src = model_path.name
    print(f"artifact={src}")
    print(f"Example left-corner 25+ ft jumper (probabilistic FG) → xShot = {prob:.3f}")
    return 0
