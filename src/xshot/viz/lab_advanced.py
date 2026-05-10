"""Force LAB / API payloads through the same advanced scalars as the sliders (post-heuristic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xshot.features.tracking_synth import contest_bucket_ix, defender_rel_angle_rad, synth_scalar_def_xy_inches


def apply_lab_advanced_overrides(feat: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    """
    After ``add_advanced_features``, re-apply explicit user tracking fields so inference
    rows never inherit hash-jitter azimuth or merged-row heuristics for contested geometry.

    Training rows without GAME_ID still use synthetic jitter inside ``add_advanced_features``;
    LAB rows set ``defender_distance_ft`` / touch / movement on the upstream dict — we mirror
    those values here and refresh bucket + relative angle from the user contest slider.
    """
    out = feat
    lx = float(pd.to_numeric(out["loc_x_ft"], errors="coerce").iloc[0])
    ly = float(pd.to_numeric(out["loc_y_ft"], errors="coerce").iloc[0])
    shx_in = lx * 12.0
    shy_in = ly * 12.0

    ddf = float(payload.get("defender_distance_ft", 4.0))
    ddf = max(0.5, ddf)
    contest_deg = float(payload.get("defender_contest_azimuth_deg", 0.0))
    dex_in, dey_in = synth_scalar_def_xy_inches(lx, ly, ddf, contest_deg)
    bucket = float(contest_bucket_ix(np.asarray([ddf], dtype=np.float64))[0])
    rel = float(defender_rel_angle_rad(np.asarray([dex_in]), np.asarray([dey_in]), np.asarray([shx_in]), np.asarray([shy_in]))[0])

    out = out.copy()
    out["defender_distance_ft"] = np.float32(ddf)
    out["def_contest_open_bucket"] = np.float32(bucket)
    out["defender_rel_angle_rad"] = np.float32(rel)
    out["defender_loc_x_inches"] = dex_in
    out["defender_loc_y_inches"] = dey_in

    out["dribbles_before_shot"] = np.float32(float(payload.get("dribbles_before_shot", 1.0)))
    out["touch_time_sec"] = np.float32(float(payload.get("touch_time_sec", 1.5)))
    out["time_since_catch"] = np.float32(float(payload.get("time_since_catch", 0.8)))
    out["distance_traveled_before_shot"] = np.float32(
        float(payload.get("distance_traveled_before_shot", 1.0))
    )
    out["rest_days_since_prev_game"] = np.float32(
        float(payload.get("rest_days_since_prev_game", 2.0))
    )
    out["is_back_to_back"] = int(bool(payload.get("is_back_to_back", False)))
    return out
