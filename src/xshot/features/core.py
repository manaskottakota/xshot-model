"""Core contextual + spatial engineered columns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    lx = out["LOC_X"].astype(float) / 10.0
    ly = out["LOC_Y"].astype(float) / 10.0
    # Angle in horizontal plane from basket outward (NBA chart: rim near origin baseline top)
    out["loc_x_ft"] = lx
    out["loc_y_ft"] = ly
    out["shot_angle_rad"] = np.arctan2(lx, np.maximum(np.abs(ly), 1e-3))

    out["shot_distance_ft"] = out["SHOT_DISTANCE"].astype(float)

    zb = out["SHOT_ZONE_BASIC"].astype(str)
    out["is_restricted_area"] = (zb == "Restricted Area").astype(np.int8)
    out["is_corner_three"] = zb.str.contains("Corner 3", case=False, na=False).astype(
        np.int8
    )
    out["is_midrange"] = (
        (~out["is_restricted_area"].astype(bool))
        & (~zb.str.contains("3", na=False))
    ).astype(np.int8)

    out["period"] = out["PERIOD"].astype(np.int8)
    mq = out["MINUTES_REMAINING"].astype(float)
    sq = out["SECONDS_REMAINING"].astype(float)
    secs_left_q = mq * 60.0 + sq
    out["secs_left_in_quarter"] = secs_left_q
    early = secs_left_q > 360  # More than six minutes remaining in this quarter
    out["early_in_quarter"] = early.astype(np.int8)

    out["shooting_team_is_home"] = out.get(
        "shooting_team_is_home", pd.Series(-1, index=out.index)
    ).astype(float)
    hs = pd.to_numeric(
        out.get("score_diff_shooting_perspective", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    out["score_diff_shooting_perspective_safe"] = hs
    out["shooting_team_ahead"] = (hs > 0).astype(np.int8)
    out["shooting_team_trailing"] = (hs < 0).astype(np.int8)

    clutch = (out["period"] >= 4) & (mq <= 5)
    out["clutch_time"] = clutch.astype(np.int8)

    st = out.get("season_type", pd.Series("Regular Season", index=out.index)).astype(
        str
    )
    out["is_playoffs"] = st.str.contains("Playoff", case=False, na=False).astype(
        np.int8
    )

    # Shot clock (typically absent here) → numeric sentinel avoids imputer warnings.
    if "SHOT_CLOCK" in out.columns:
        sc = pd.to_numeric(out["SHOT_CLOCK"], errors="coerce")
    else:
        sc = pd.Series(np.nan, index=out.index)
    unknown = ~sc.notna()
    out["shot_clock_remaining"] = sc.fillna(-1.0)
    out["shot_clock_known"] = (~unknown).astype(np.int8)

    act = out["ACTION_TYPE"].astype(str)
    low = act.str.lower()
    out["shot_style_layup"] = low.str.contains("layup", na=False).astype(np.int8)
    out["shot_style_dunk"] = low.str.contains("dunk", na=False).astype(np.int8)
    out["shot_style_hook"] = low.str.contains("hook", na=False).astype(np.int8)
    out["shot_style_fadeaway"] = low.str.contains("fade", na=False).astype(np.int8)
    out["shot_style_jumper"] = low.str.contains("jump shot", na=False).astype(np.int8)
    out["shot_style_pullup"] = low.str.contains("pull", na=False).astype(np.int8)
    out["shot_style_stepback"] = low.str.contains("step", na=False).astype(np.int8)

    stype = out["SHOT_TYPE"].astype(str)
    out["is_three"] = stype.str.contains("3PT", na=False).astype(np.int8)

    return out
