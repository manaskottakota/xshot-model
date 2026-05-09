"""League shot chart ingestion (team aggregate, player_id=0)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams

from xshot.ingest.cache import ensure_dir, throttle

_LAST = [0.0]
_MIN_INTERVAL = 0.65


def _season_id_from_string(season: str) -> str:
    """Map '2023-24' -> '22023' used by some NBA endpoints (not all)."""
    return season  # ShotChartDetail uses season_nullable like '2023-24'


def fetch_shots_season(
    season: str,
    season_type: str = "Regular Season",
    out_path: Path | None = None,
    team_ids: list[int] | None = None,
    rate_limit_s: float = _MIN_INTERVAL,
) -> pd.DataFrame:
    """
    Fetch all FGA rows for ``season`` by iterating NBA teams with player_id=0
    (team shot aggregate). Writes ``out_path`` as Parquet when provided.
    """
    ensure_dir(Path("data/raw"))
    tids = team_ids if team_ids is not None else [t["id"] for t in teams.get_teams()]
    chunks: list[pd.DataFrame] = []
    for tid in tids:
        throttle(_LAST, rate_limit_s)
        scd = shotchartdetail.ShotChartDetail(
            team_id=int(tid),
            player_id=0,
            season_nullable=_season_id_from_string(season),
            context_measure_simple="FGA",
            season_type_all_star=season_type,
            timeout=60,
        )
        df = scd.get_data_frames()[0].copy()
        df["season"] = season
        df["season_type"] = season_type
        chunks.append(df)
        time.sleep(0.05)

    shots = pd.concat(chunks, ignore_index=True)
    shots = shots.drop_duplicates(
        subset=["GAME_ID", "GAME_EVENT_ID", "TEAM_ID", "PLAYER_ID"], keep="first"
    )

    if out_path is not None:
        ensure_dir(out_path.parent)
        shots.to_parquet(out_path, index=False)

    return shots
