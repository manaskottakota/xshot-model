"""Assemble one training-aligned feature row from structured inference inputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from xshot.datasets import PRIOR_LAST_N_FG_COL, assert_no_forbidden, feature_columns
from xshot.features.advanced import add_advanced_features
from xshot.features.core import add_core_features
from xshot.features.tracking_synth import synth_scalar_def_xy_inches

_REPO_ROOT = Path(__file__).resolve().parents[3]


def archetype_action_type(archetype: str) -> str:
    a = archetype.lower().strip()
    mapping = {
        "layup": "Driving Layup Shot",
        "dunk": "Driving Dunk Shot",
        "hook": "Hook Shot",
        "fadeaway": "Fadeaway Jump Shot",
        "jumper": "Jump Shot",
        "pullup": "Pullup Jump shot",
        "stepback": "Step Back Jump shot",
        "floater": "Driving Floating Jump Shot",
    }
    raw = mapping.get(a, mapping["jumper"])
    return raw.replace("#", "").strip()


def infer_shot_zone_basic(lx_ft: float, ly_ft: float, is_three: bool) -> str:
    ax = abs(lx_ft)
    dist = math.hypot(lx_ft, ly_ft)

    if not is_three:
        if dist < 5.25:
            return "Restricted Area"
        if ax < 11.75 and ly_ft <= 21.75:
            return "In The Paint (Non-RA)"
        return "Mid-Range"

    if ax >= 20.25 and ly_ft <= 38.75:
        return "Left Corner 3" if lx_ft < 0 else "Right Corner 3"
    return "Above the Break 3"


def player_prior_overrides(profile: str) -> dict[str, float]:
    presets: dict[str, dict[str, float]] = {
        "league_average": {
            "prior_cum_fg_pct": 0.447,
            "prior_three_attempt_share": 0.42,
            "prior_attempts_global": 520.0,
            PRIOR_LAST_N_FG_COL: 0.44,
        },
        "elite_spot_up": {
            "prior_cum_fg_pct": 0.491,
            "prior_three_attempt_share": 0.58,
            "prior_attempts_global": 620.0,
            PRIOR_LAST_N_FG_COL: 0.48,
        },
        "rim_finisher_big": {
            "prior_cum_fg_pct": 0.582,
            "prior_three_attempt_share": 0.11,
            "prior_attempts_global": 980.0,
            PRIOR_LAST_N_FG_COL: 0.59,
        },
        "combo_guard": {
            "prior_cum_fg_pct": 0.459,
            "prior_three_attempt_share": 0.47,
            "prior_attempts_global": 715.0,
            PRIOR_LAST_N_FG_COL: 0.45,
        },
    }
    base = presets.get(profile, presets["league_average"])
    return dict(base)


def inference_X_from_payload(
    payload: dict[str, Any],
    *,
    features: Literal["core", "core+advanced"],
) -> pd.DataFrame:
    advanced = features == "core+advanced"
    lx = float(payload["loc_x_ft"])
    ly = float(payload["loc_y_ft"])
    is_three = bool(payload.get("is_three", False))

    archetype = str(payload.get("shot_archetype", "jumper"))
    zone = payload.get("shot_zone_basic_override")
    zb = infer_shot_zone_basic(lx, ly, is_three) if not zone else str(zone)

    pd_ = max(1, min(8, int(payload.get("period", 1))))
    mq = float(payload.get("minutes_remaining", 10.0))
    sq = float(payload.get("seconds_remaining", 0.0))

    sd = payload.get("shot_distance_ft")
    if sd is None or (isinstance(sd, str) and not sd.strip()):
        sd = math.hypot(lx, ly)
    sd = float(sd)

    sc_raw = payload.get("shot_clock_seconds")
    sc_known = bool(payload.get("shot_clock_known", False))
    if sc_known and sc_raw is not None:
        sc_val = float(sc_raw)
    else:
        sc_val = np.nan

    scoring = pd.to_numeric(
        pd.Series([float(payload.get("score_diff", 0.0))]), errors="coerce"
    ).fillna(0.0)

    shot_type_label = "3PT Field Goal" if is_three else "2PT Field Goal"

    home = payload.get("shooting_team_home")
    if isinstance(home, bool):
        sth = float(1 if home else 0)
    elif home is None:
        sth = 1.0
    else:
        sth = float(home)

    row: dict[str, Any] = {
        "LOC_X": lx * 10.0,
        "LOC_Y": ly * 10.0,
        "SHOT_DISTANCE": sd,
        "SHOT_ZONE_BASIC": zb,
        "SHOT_ZONE_AREA": "Center(C)",
        "SHOT_ZONE_RANGE": "",
        "PERIOD": pd_,
        "MINUTES_REMAINING": mq,
        "SECONDS_REMAINING": sq,
        "ACTION_TYPE": archetype_action_type(archetype),
        "SHOT_TYPE": shot_type_label,
        "SHOT_CLOCK": sc_val,
        "season_type": "Playoffs" if payload.get("is_playoffs") else "Regular Season",
        "shooting_team_is_home": sth,
        "score_diff_shooting_perspective": float(scoring.iloc[0]),
    }

    if advanced:
        ddf = float(payload.get("defender_distance_ft", 4.0))
        ddf = max(0.5, ddf)
        contest_deg = float(payload.get("defender_contest_azimuth_deg", 0.0))
        dex_in, dey_in = synth_scalar_def_xy_inches(lx, ly, ddf, contest_deg)
        row["defender_distance_ft"] = ddf
        row["defender_loc_x_inches"] = dex_in
        row["defender_loc_y_inches"] = dey_in
        row["dribbles_before_shot"] = float(payload.get("dribbles_before_shot", 1.0))
        row["touch_time_sec"] = float(payload.get("touch_time_sec", 1.5))
        row["rest_days_since_prev_game"] = float(payload.get("rest_days_since_prev_game", 2.0))
        row["is_back_to_back"] = int(bool(payload.get("is_back_to_back", False)))
        row["time_since_catch"] = float(payload.get("time_since_catch", 0.8))
        row["distance_traveled_before_shot"] = float(
            payload.get("distance_traveled_before_shot", 1.0)
        )
        row["player_velocity_x"] = np.nan
        row["player_velocity_y"] = np.nan
        row["defender_velocity_x"] = np.nan
        row["defender_velocity_y"] = np.nan

    df0 = pd.DataFrame([row])
    feat = add_core_features(df0)
    feat["shot_zone_basic"] = feat["SHOT_ZONE_BASIC"].astype(str)

    prof = str(payload.get("player_profile", "league_average"))
    pri = player_prior_overrides(prof)
    for k, v in pri.items():
        feat[k] = float(v)

    if advanced:
        feat = add_advanced_features(feat)

    num, cat = feature_columns(features)
    cols = num + cat
    missing_cols = set(cols) - set(feat.columns)
    if missing_cols:
        raise KeyError(f"Internal feature assemble missing columns {sorted(missing_cols)}")
    raw_x = feat[cols].copy()
    assert_no_forbidden(raw_x)
    return raw_x


def default_model_search_paths() -> list[Path]:
    rel = (
        Path("artifacts/run_default/xshot_primary_calibrated.joblib"),
        Path("artifacts/run_default/xshot_primary.joblib"),
        Path("artifacts/run_default/histogram_gradient_boosting.joblib"),
        Path("artifacts/run_default/xgboost.joblib"),
    )
    out: list[Path] = []
    for r in rel:
        out.extend((r, _REPO_ROOT / r))
    return out


def resolve_model_path(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.exists():
        return explicit
    seen: set[Path] = set()
    for p in default_model_search_paths():
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.exists():
            return rp
    return None
