"""Defender geometry, fatigue, touches — merges optional tracking CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xshot.ingest.pbp import shot_elapsed_seconds_in_game


def add_advanced_features(shots_core: pd.DataFrame) -> pd.DataFrame:
    out = shots_core.copy()

    out["defender_distance_ft"] = pd.to_numeric(
        out.get("defender_distance_ft", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    buckets = pd.cut(
        out["defender_distance_ft"],
        bins=[-np.inf, 2.5, 4.5, 6.5, np.inf],
        labels=[0.0, 1.0, 2.0, 3.0],
    )
    codes = buckets.cat.codes.astype(float)
    codes[codes < 0] = np.nan
    out["def_contest_open_bucket"] = codes

    out["defender_loc_x_inches"] = pd.to_numeric(
        out.get("defender_loc_x_inches", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["defender_loc_y_inches"] = pd.to_numeric(
        out.get("defender_loc_y_inches", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    # Coarse planar angle between shooter rim vector and shooter-defender delta
    shx_in = out["loc_x_ft"].astype(float) * 12.0
    shy_in = out["loc_y_ft"].astype(float) * 12.0
    dex_in = out["defender_loc_x_inches"].astype(float)
    dey_in = out["defender_loc_y_inches"].astype(float)
    vec_x = dex_in - shx_in
    vec_y = dey_in - shy_in
    out["defender_rel_angle_rad"] = np.arctan2(vec_x, np.maximum(np.abs(vec_y), 1e-3))
    out["defender_geom_known"] = out["defender_loc_x_inches"].notna().astype(np.int8)

    out["dribbles_before_shot"] = pd.to_numeric(
        out.get("dribbles_before_shot", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["touch_time_sec"] = pd.to_numeric(
        out.get("touch_time_sec", pd.Series(np.nan, index=out.index)), errors="coerce"
    )

    elapsed = shot_elapsed_seconds_in_game(
        out["period"], out["MINUTES_REMAINING"], out["SECONDS_REMAINING"]
    )
    out["elapsed_game_sec_approx"] = elapsed.astype(float)
    out["player_load_game_min_approx"] = elapsed.astype(float) / 60.0

    rd = pd.to_numeric(
        out.get("rest_days_since_prev_game", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["rest_days_since_prev_game"] = rd
    out["is_back_to_back"] = out.get(
        "is_back_to_back", pd.Series(0, index=out.index)
    ).astype(int)

    return out
