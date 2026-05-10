"""Defender geometry, fatigue, touches — optional SportVU merge + deterministic fills."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xshot.features.tracking_synth import (
    contest_azimuth_from_u,
    contest_bucket_ix,
    defender_rel_angle_rad,
    defender_xy_inches_batch,
    heuristic_def_distance_ft,
    stable_jitter01,
)
from xshot.ingest.pbp import shot_elapsed_seconds_in_game


def _tracking_identity_keys(df: pd.DataFrame) -> pd.DataFrame:
    gid = df.get("GAME_ID")
    ge = df.get("GAME_EVENT_ID")
    if gid is not None and ge is not None:
        frame = pd.DataFrame({"GAME_ID": gid.astype(str), "GAME_EVENT_ID": ge.astype(str)})
        pid = df.get("PLAYER_ID")
        if pid is not None:
            frame["PLAYER_ID"] = pid.astype(str)
        return frame
    return pd.DataFrame({"row_id": np.arange(len(df), dtype=np.int64).astype(str)})


def add_advanced_features(shots_core: pd.DataFrame) -> pd.DataFrame:
    out = shots_core.copy()
    lx = pd.to_numeric(out["loc_x_ft"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ly = pd.to_numeric(out["loc_y_ft"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    lx_abs = np.abs(lx)
    sd = pd.to_numeric(out["shot_distance_ft"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    thr_mask = pd.to_numeric(out["is_three"], errors="coerce").fillna(0).to_numpy() > 0.5

    keys = _tracking_identity_keys(out)
    u1 = stable_jitter01(keys, salt=0xA11CE5EEDBADF00D).astype(np.float64)
    u2 = stable_jitter01(keys, salt=0x5EEDEADBEEFACA11).astype(np.float64)

    dist_merged = pd.to_numeric(
        out.get("defender_distance_ft", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    dex_obs = pd.to_numeric(
        out.get("defender_loc_x_inches", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    dey_obs = pd.to_numeric(
        out.get("defender_loc_y_inches", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    drib_m = pd.to_numeric(
        out.get("dribbles_before_shot", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    touch_m = pd.to_numeric(
        out.get("touch_time_sec", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    tsc_m = pd.to_numeric(
        out.get("time_since_catch", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    trv_m = pd.to_numeric(
        out.get("distance_traveled_before_shot", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    heuristic_dist = heuristic_def_distance_ft(sd, lx_abs, u1)

    filled_dist = np.where(np.isfinite(dist_merged), dist_merged, heuristic_dist)

    psi_deg = contest_azimuth_from_u(u2)

    dex_fill, dey_fill = defender_xy_inches_batch(lx, ly, filled_dist, psi_deg)
    need_geom = (~np.isfinite(dex_obs)) | (~np.isfinite(dey_obs))
    dex_used = np.where(need_geom, dex_fill, dex_obs)
    dey_used = np.where(need_geom, dey_fill, dey_obs)

    base_db = np.where(thr_mask, 0.62, 1.92)
    db_heuristic = np.clip(base_db + (u1 - 0.55) * 0.92, 0.0, 24.5)
    filled_db = np.where(np.isfinite(drib_m), drib_m, db_heuristic)

    base_touch = np.where(thr_mask, 1.12, 2.05) + filled_db * 0.42
    touch_heuristic = np.clip(base_touch + (u2 - 0.52) * 0.74, 0.25, 22.8)
    filled_touch = np.where(np.isfinite(touch_m), touch_m, touch_heuristic)

    tsc_heuristic = np.clip(0.12 + filled_touch * 0.62 + u1 * 0.22, 0.05, 18.9)
    filled_tsc = np.where(np.isfinite(tsc_m), tsc_m, tsc_heuristic)

    trv_heuristic = np.clip(0.86 + filled_db * 0.71 + lx_abs * 0.018 + u2 * 0.61, 0.1, 90.5)
    filled_trv = np.where(np.isfinite(trv_m), trv_m, trv_heuristic)

    buckets = contest_bucket_ix(filled_dist)
    if not np.all(np.isfinite(buckets)):
        raise ValueError("def_contest_open_bucket unresolved — check defender_distance pipeline")

    shx_in = lx * 12.0
    shy_in = ly * 12.0
    rel_ang = defender_rel_angle_rad(dex_used, dey_used, shx_in, shy_in)

    out["defender_distance_ft"] = filled_dist.astype(np.float32)
    out["def_contest_open_bucket"] = buckets.astype(np.float32)
    out["defender_rel_angle_rad"] = rel_ang.astype(np.float32)
    # Retain merged / synthetic planar inches for debugging — excluded from ACTIVE_X slices.
    out["defender_loc_x_inches"] = dex_used.astype(np.float64)
    out["defender_loc_y_inches"] = dey_used.astype(np.float64)

    elapsed = shot_elapsed_seconds_in_game(
        out["period"], out["MINUTES_REMAINING"], out["SECONDS_REMAINING"]
    )
    out["elapsed_game_sec_approx"] = elapsed.astype(float)
    out["player_load_game_min_approx"] = elapsed.astype(float) / 60.0

    out["dribbles_before_shot"] = filled_db.astype(np.float32)
    out["touch_time_sec"] = filled_touch.astype(np.float32)
    out["time_since_catch"] = filled_tsc.astype(np.float32)
    out["distance_traveled_before_shot"] = filled_trv.astype(np.float32)

    rd_np = pd.to_numeric(
        out.get("rest_days_since_prev_game", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    rest_syn = np.clip(2.3 + (u2 - 0.52) * 2.95, 0.0, 360.0)
    out["rest_days_since_prev_game"] = np.where(np.isfinite(rd_np), rd_np, rest_syn).astype(
        np.float32
    )

    out["is_back_to_back"] = out.get(
        "is_back_to_back", pd.Series(0, index=out.index)
    ).astype(int)

    # Latent kinematics placeholders (inactive in ACTIVE_X ADVANCED_FEATURES).
    for col in (
        "player_velocity_x",
        "player_velocity_y",
        "defender_velocity_x",
        "defender_velocity_y",
        "shooter_speed",
        "defender_speed",
    ):
        if col not in out.columns:
            out[col] = np.nan

    return out
