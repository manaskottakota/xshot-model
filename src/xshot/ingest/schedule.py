"""Back-to-back and rest proxies from GAME_DATE streaks."""

from __future__ import annotations

import pandas as pd


def enrich_shots_schedule_fatigue(shots: pd.DataFrame) -> pd.DataFrame:
    """
    For each TEAM_ID + GAME_DATE, compute days since that team's prior game.

    Columns added: ``games_since_rest_days`` (NaN for first game in data),
    ``is_back_to_back`` (1 if previous game was the prior calendar day).
    """
    df = shots.copy()
    if "GAME_DATE" not in df.columns:
        raise ValueError("shots must include GAME_DATE")

    df["_game_dt"] = pd.to_datetime(df["GAME_DATE"].astype(str), format="%Y%m%d")
    tpl = df[["TEAM_ID", "GAME_ID", "_game_dt"]].drop_duplicates()
    tpl = tpl.sort_values(["TEAM_ID", "_game_dt", "GAME_ID"])
    tpl["prev_dt"] = tpl.groupby("TEAM_ID")["_game_dt"].shift(1)
    tpl["rest_days_since_prev_game"] = (
        tpl["_game_dt"] - tpl["prev_dt"]
    ).dt.days - 1
    tpl["is_back_to_back"] = (
        tpl["_game_dt"] - tpl["prev_dt"]).dt.days == 1

    keyed = tpl.set_index(["TEAM_ID", "GAME_ID"])
    df = df.merge(
        keyed[["rest_days_since_prev_game", "is_back_to_back"]],
        left_on=["TEAM_ID", "GAME_ID"],
        right_index=True,
        how="left",
    )
    df = df.drop(columns=["_game_dt"], errors="ignore")
    df["rest_days_since_prev_game"] = df["rest_days_since_prev_game"].astype(float)
    df["is_back_to_back"] = df["is_back_to_back"].fillna(False).astype(int)
    return df
