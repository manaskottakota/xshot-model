"""Assemble augmented shot tables → model-ready ``X, y``."""

from __future__ import annotations

from pathlib import Path
from typing import Collection, Literal

import numpy as np
import pandas as pd

from xshot.ingest.pbp import enrich_shots_with_pbp
from xshot.ingest.schedule import enrich_shots_schedule_fatigue
from xshot.ingest.shots import fetch_shots_season
from xshot.ingest.tracking import merge_tracking_csv


ROLLING_LAST_N_GAMES = 10


def load_or_fetch_shots_cached(
    season: str,
    *,
    season_type: str = "Regular Season",
    cache_shots: Path | None = None,
    team_ids: list[int] | None = None,
) -> pd.DataFrame:
    if cache_shots is not None and cache_shots.exists():
        return pd.read_parquet(cache_shots)
    return fetch_shots_season(
        season,
        season_type=season_type,
        out_path=cache_shots,
        team_ids=team_ids,
    )


def merge_all_context(
    shots: pd.DataFrame,
    *,
    tracking_path: Path | None = None,
    pbp_max_games: int | None = None,
) -> pd.DataFrame:
    df = enrich_shots_with_pbp(shots, max_games=pbp_max_games)
    df = enrich_shots_schedule_fatigue(df)
    df = merge_tracking_csv(df, tracking_path)
    return df


def build_feature_table(
    merged: pd.DataFrame,
    features: Literal["core", "core+advanced"],
) -> pd.DataFrame:
    from xshot.features.advanced import add_advanced_features
    from xshot.features.core import add_core_features
    from xshot.features.rolling import add_rolling_player_features

    out = add_core_features(merged)
    out["shot_zone_basic"] = out["SHOT_ZONE_BASIC"].astype(str)
    out = add_rolling_player_features(out, last_n_games=ROLLING_LAST_N_GAMES)
    if features == "core+advanced":
        out = add_advanced_features(out)
    return out


FORBIDDEN_IN_X = frozenset(
    {
        "PLAYER_NAME",
        "TEAM_NAME",
        "PLAYER_ID",
        "TEAM_ID",
        "GAME_ID",
        "GAME_DATE",
        "GAME_EVENT_ID",
        "HTM",
        "VTM",
        "GRID_TYPE",
        "EVENT_TYPE",
        "ACTION_TYPE",
        "SHOT_TYPE",
        "SHOT_ZONE_BASIC",
        "SHOT_ZONE_AREA",
        "SHOT_ZONE_RANGE",
        "season_type",
        "pbp_clock_iso",
        "pbp_location",
        "HOME_TEAM_ID",
        "VISITOR_TEAM_ID",
    }
)


PRIOR_LAST_N_FG_COL = f"prior_last_{ROLLING_LAST_N_GAMES}g_fg_pct"

NUMERIC_CORE_FEATURES = [
    "loc_x_ft",
    "loc_y_ft",
    "shot_angle_rad",
    "shot_distance_ft",
    "abs_loc_x_ft",
    "lateral_distance_ratio",
    "fourth_quarter_clock_pressure",
    # Reversible omission: duplicated signal vs continuous score-diff clip
    # "shooting_team_ahead",
    # "shooting_team_trailing",
    "score_diff_normalized",
    "is_restricted_area",
    "is_corner_three",
    "is_midrange",
    "period",
    "secs_left_in_quarter",
    "early_in_quarter",
    "shooting_team_is_home",
    "score_diff_shooting_perspective_safe",
    "clutch_time",
    "is_playoffs",
    "shot_clock_remaining",
    "shot_clock_known",
    "shot_style_layup",
    "shot_style_dunk",
    "shot_style_hook",
    "shot_style_fadeaway",
    "shot_style_jumper",
    "shot_style_pullup",
    "shot_style_stepback",
    "is_three",
    "prior_cum_fg_pct",
    "prior_three_attempt_share",
    "prior_attempts_global",
    PRIOR_LAST_N_FG_COL,
    # Game-state interactions (spatial / clock context crosses)
    "clutch_three_interaction",
    "trailing_pressure_three",
]

ADVANCED_NUMERIC_FEATURES = [
    "defender_distance_ft",
    "def_contest_open_bucket",
    "defender_rel_angle_rad",
    "dribbles_before_shot",
    "touch_time_sec",
    "elapsed_game_sec_approx",
    "player_load_game_min_approx",
    "rest_days_since_prev_game",
    "is_back_to_back",
    # Kinematics when ingest attaches velocity parquet (match training merges):
    # "shooter_speed",
    # "defender_speed",
    "time_since_catch",
    "distance_traveled_before_shot",
]

FEATURE_ABLATION_GROUPS = {
    "spatial_context": frozenset(
        {
            "loc_x_ft",
            "loc_y_ft",
            "shot_angle_rad",
            "shot_distance_ft",
            "abs_loc_x_ft",
            "lateral_distance_ratio",
            "fourth_quarter_clock_pressure",
            "is_restricted_area",
            "is_corner_three",
            "is_midrange",
            "is_three",
        },
    ),
    "game_state_pressure": frozenset(
        {
            "score_diff_normalized",
            "score_diff_shooting_perspective_safe",
            "shooting_team_is_home",
            "secs_left_in_quarter",
            "early_in_quarter",
            "clutch_time",
            "period",
            "clutch_three_interaction",
            "fourth_quarter_clock_pressure",
            "trailing_pressure_three",
        },
    ),
    "prior_and_style": frozenset(
        {
            "prior_cum_fg_pct",
            "prior_three_attempt_share",
            "prior_attempts_global",
            PRIOR_LAST_N_FG_COL,
            "shot_style_layup",
            "shot_style_dunk",
            "shot_style_hook",
            "shot_style_fadeaway",
            "shot_style_jumper",
            "shot_style_pullup",
            "shot_style_stepback",
            "shot_clock_remaining",
            "shot_clock_known",
            "is_playoffs",
        },
    ),
}


def resolve_ablation_union(group_names: list[str]) -> frozenset[str]:
    out: list[str] = []
    unk: list[str] = []
    for g in group_names:
        fg = FEATURE_ABLATION_GROUPS.get(g)
        if fg is None:
            unk.append(g)
        else:
            out.extend(fg)
    if unk:
        raise KeyError(f"Unknown ablation group(s): {unk}. Known: {sorted(FEATURE_ABLATION_GROUPS.keys())}")
    return frozenset(out)


def assert_no_forbidden(df: pd.DataFrame) -> None:
    present = sorted(FORBIDDEN_IN_X.intersection(df.columns))
    if present:
        raise ValueError(f"Forbidden identity/raw columns in X: {present}")


def feature_columns(
    mode: Literal["core", "core+advanced"],
    omit_features: Collection[str] | None = None,
) -> tuple[list[str], list[str]]:
    omit = frozenset(omit_features or ())
    cat = [c for c in ["shot_zone_basic"] if c not in omit]
    num_raw = list(NUMERIC_CORE_FEATURES)
    if mode == "core+advanced":
        num_raw = num_raw + ADVANCED_NUMERIC_FEATURES
    num = [c for c in num_raw if c not in omit]
    return num, cat


def X_y_from_table(
    table: pd.DataFrame,
    features: Literal["core", "core+advanced"],
    omit_features: Collection[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    num, cat = feature_columns(features, omit_features=omit_features)
    cols = num + cat
    missing = set(cols) - set(table.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {sorted(missing)}")
    raw_x = table[cols].copy()
    assert_no_forbidden(raw_x)
    y = (
        pd.to_numeric(table["SHOT_MADE_FLAG"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    return raw_x, y


def training_manifest_row(
    *,
    seasons: list[str],
    n_rows: int,
    features: str,
    pbp_max_games: int | None,
    tracking_path: str | None,
) -> dict:
    return {
        "seasons": seasons,
        "n_rows": n_rows,
        "features": features,
        "pbp_max_games": pbp_max_games,
        "tracking_path": tracking_path,
        "rolling_last_n_games": ROLLING_LAST_N_GAMES,
    }
