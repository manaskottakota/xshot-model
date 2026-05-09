"""Optional SportVU-like per-shot CSV/Parquet merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRACKING_COLUMNS = [
    "defender_distance_ft",
    "dribbles_before_shot",
    "touch_time_sec",
    "defender_loc_x_inches",
    "defender_loc_y_inches",
]


def merge_tracking_csv(
    shots: pd.DataFrame, path: Path | None = None
) -> pd.DataFrame:
    """
    Merge optional tracking file keyed by ``GAME_ID`` + ``GAME_EVENT_ID``
    (+ ``PLAYER_ID`` when present on both frames).

    Aliases normalized from common NBA-derived exports::

        CLOSE_DEF_DIST -> defender_distance_ft
        DRIBBLES -> dribbles_before_shot
        TOUCH_TIME -> touch_time_sec
        CLOSEDEF_{X,Y} -> defender_loc_{x,y}_inches
    """
    out = shots.copy()
    for c in TRACKING_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA

    if path is None or not Path(path).exists():
        out["tracking_merge_ok"] = 0
        return out

    tr_path = Path(path)
    tr = pd.read_parquet(tr_path) if tr_path.suffix == ".parquet" else pd.read_csv(tr_path)
    colmap = {
        "CLOSE_DEF_DIST": "defender_distance_ft",
        "DRIBBLES": "dribbles_before_shot",
        "TOUCH_TIME": "touch_time_sec",
        "CLOSEDEF_X": "defender_loc_x_inches",
        "CLOSEDEF_Y": "defender_loc_y_inches",
    }
    rename = {k: v for k, v in colmap.items() if k in tr.columns}
    tr = tr.rename(columns=rename)

    keys = ["GAME_ID", "GAME_EVENT_ID"]
    if "PLAYER_ID" in out.columns and "PLAYER_ID" in tr.columns:
        keys.append("PLAYER_ID")

    extra_cols = [c for c in TRACKING_COLUMNS if c in tr.columns]
    tr_slim = tr[keys + extra_cols].drop_duplicates(keys)
    merged = out.merge(tr_slim, how="left", on=keys, suffixes=("", "_trk"))

    for base in TRACKING_COLUMNS:
        trk_col = f"{base}_trk"
        if trk_col in merged.columns:
            merged[base] = merged[trk_col].combine_first(merged[base])
            merged.drop(columns=[trk_col], inplace=True)

    merged["tracking_merge_ok"] = merged["defender_distance_ft"].notna().astype(int)
    return merged
