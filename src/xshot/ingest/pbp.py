"""Play-by-play V3 enrichment (score differential, possession side)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

from xshot.ingest.cache import ensure_dir, throttle

_LAST = [0.0]


def fetch_playbyplay_v3(game_id: str, *, timeout: int = 60) -> pd.DataFrame:
    throttle(_LAST, 0.6)
    pbp = playbyplayv3.PlayByPlayV3(game_id=str(game_id), timeout=timeout)
    return pbp.get_data_frames()[0].copy()


def _prep_pbp_for_merge(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.sort_values(["gameId", "actionNumber"]).copy()
    grp = df.groupby("gameId", sort=False)

    df["_sh"] = grp["scoreHome"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").ffill()
    )
    df["_sa"] = grp["scoreAway"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").ffill()
    )
    # Start of game rows before scoring: propagate next known backwards
    df["_sh"] = grp["_sh"].transform(lambda s: s.bfill()).fillna(0).astype(int)
    df["_sa"] = grp["_sa"].transform(lambda s: s.bfill()).fillna(0).astype(int)

    loc_up = grp["location"].transform(lambda s: s.ffill().bfill())
    df["_location_ff"] = loc_up
    df["shooting_team_is_home"] = (df["_location_ff"] == "h").astype(np.int8)
    shooter_home = df["shooting_team_is_home"].astype(bool)
    df["_team_pts"] = np.where(shooter_home, df["_sh"], df["_sa"])
    df["_opp_pts"] = np.where(shooter_home, df["_sa"], df["_sh"])

    rename = {
        "_sh": "pbp_score_home",
        "_sa": "pbp_score_away",
        "_location_ff": "pbp_location",
        "_team_pts": "pbp_team_score",
        "_opp_pts": "pbp_opp_score",
        "period": "pbp_period",
        "clock": "pbp_clock_iso",
        "location": "pbp_location_raw",
    }
    out = df.rename(columns=rename)
    out["pbp_team_spread"] = out["pbp_team_score"] - out["pbp_opp_score"]
    keep = [
        "gameId",
        "actionNumber",
        "pbp_score_home",
        "pbp_score_away",
        "pbp_team_score",
        "pbp_opp_score",
        "pbp_team_spread",
        "shooting_team_is_home",
        "pbp_location",
        "pbp_period",
        "pbp_clock_iso",
    ]
    return out[keep]


def enrich_shots_with_pbp(
    shots: pd.DataFrame,
    cache_dir: Path = Path("data/raw/pbp"),
    *,
    max_games: int | None = None,
    rate_limit_s: float = 0.6,
) -> pd.DataFrame:
    """Left-merge PBP-derived context onto shot rows keyed by GAME_ID + GAME_EVENT_ID."""
    global _LAST
    _LAST[0] = 0.0
    ensure_dir(cache_dir)

    gid_list = shots["GAME_ID"].astype(str).unique().tolist()
    if max_games is not None:
        gid_list = gid_list[:max_games]

    pbp_chunks: list[pd.DataFrame] = []
    for gid in gid_list:
        fp = cache_dir / f"pbp_{gid}.parquet"
        if fp.exists():
            raw = pd.read_parquet(fp)
        else:
            throttle(_LAST, rate_limit_s)
            raw = fetch_playbyplay_v3(gid)
            ensure_dir(cache_dir.parent)
            raw.to_parquet(fp, index=False)
            time.sleep(0.02)
        pbp_chunks.append(_prep_pbp_for_merge(raw))

    pbp_all = pd.concat(pbp_chunks, ignore_index=True)
    merged = shots.merge(
        pbp_all,
        how="left",
        left_on=["GAME_ID", "GAME_EVENT_ID"],
        right_on=["gameId", "actionNumber"],
    )
    merged = merged.drop(columns=["gameId", "actionNumber"], errors="ignore")
    merged["shooting_team_is_home"] = merged["shooting_team_is_home"].fillna(-1)

    merged["score_diff_shooting_perspective"] = merged["pbp_team_spread"]
    merged["ahead_by_points"] = (merged["pbp_team_spread"] > 0).astype("Int64")
    return merged


def shot_elapsed_seconds_in_game(period, minutes_remaining, seconds_remaining) -> pd.Series:
    secs_left_q = minutes_remaining.astype(float) * 60.0 + seconds_remaining.astype(float)
    secs_elapsed_q = 12 * 60 - secs_left_q
    elapsed = (period.astype(float) - 1.0) * 12 * 60 + secs_elapsed_q
    return elapsed
