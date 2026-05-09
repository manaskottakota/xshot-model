"""Shifted shooter history — excludes current-shot label by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

GLOBAL_PRIOR_RATE = 0.45


def add_rolling_player_features(
    df: pd.DataFrame,
    *,
    last_n_games: int = 10,
    global_prior: float = GLOBAL_PRIOR_RATE,
) -> pd.DataFrame:
    """
    Per ``PLAYER_ID`` (ETL-only key), chronologically sorted:

    - ``prior_cum_fg_pct`` — cumulative FG% before this outcome.
    - ``prior_three_attempt_share`` — share of prior FGA that were 3PA.
    - ``prior_last{N}g_fg_pct`` — mean per-game FG% over last N **completed prior**
      games within the player's history.

    Shooting rows only; ``PLAYER_ID`` must not leak into downstream ``X``.
    """
    if "SHOT_MADE_FLAG" not in df.columns:
        raise ValueError("need SHOT_MADE_FLAG")

    out = df.copy()
    out["_game_dt"] = pd.to_datetime(out["GAME_DATE"].astype(str), format="%Y%m%d")
    out = out.sort_values(
        ["PLAYER_ID", "_game_dt", "season", "GAME_ID", "GAME_EVENT_ID"],
        kind="mergesort",
    )

    pid = out["PLAYER_ID"].astype(np.int64)
    made = pd.to_numeric(out["SHOT_MADE_FLAG"], errors="coerce").fillna(0.0)
    three = pd.to_numeric(
        out.get("is_three", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)

    cs_made = made.groupby(pid, sort=False).cumsum()
    out["prior_attempts_global"] = pid.groupby(pid, sort=False).cumcount()

    denom = np.maximum(out["prior_attempts_global"].to_numpy(dtype=float), 1.0)
    prior_c_m = cs_made.to_numpy(dtype=float) - made.to_numpy(dtype=float)
    out["prior_cum_fg_pct"] = np.where(
        out["prior_attempts_global"].to_numpy() == 0,
        global_prior,
        prior_c_m / denom,
    )

    cs_three = three.groupby(pid, sort=False).cumsum()
    prior_c_three = cs_three.to_numpy(dtype=float) - three.to_numpy(dtype=float)
    out["prior_three_attempt_share"] = np.where(
        out["prior_attempts_global"].to_numpy() == 0,
        0.35,
        prior_c_three / denom,
    )

    win_col = f"prior_last_{last_n_games}g_fg_pct"
    gsum = (
        out.assign(make=made, attempted=1.0)
        .groupby(["PLAYER_ID", "GAME_ID"], sort=False)
        .agg(make_sum=("make", "sum"), attempts=("attempted", "sum"), gm_dt=("_game_dt", "first"))
        .reset_index()
    )
    gsum["gm_fg_pct"] = gsum["make_sum"] / np.maximum(gsum["attempts"].to_numpy(), 1.0)

    buckets: list[pd.DataFrame] = []
    for _, grp in gsum.groupby("PLAYER_ID", sort=False):
        grp_sorted = grp.sort_values(["gm_dt", "GAME_ID"], kind="mergesort").reset_index(
            drop=True
        )
        rolling_mean = grp_sorted["gm_fg_pct"].shift(1).rolling(
            last_n_games, min_periods=1
        ).mean()
        buckets.append(
            grp_sorted.assign(**{win_col: rolling_mean})[
                ["PLAYER_ID", "GAME_ID", win_col]
            ]
        )

    window_df = pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame(
        columns=["PLAYER_ID", "GAME_ID", win_col]
    )
    out = out.merge(window_df, on=["PLAYER_ID", "GAME_ID"], how="left")
    out[win_col] = pd.to_numeric(out[win_col], errors="coerce")
    miss_win = out[win_col].isna()
    out.loc[miss_win, win_col] = (
        pd.to_numeric(out.loc[miss_win, "prior_cum_fg_pct"], errors="coerce").fillna(
            global_prior
        )
    )

    out.drop(columns=["_game_dt"], inplace=True, errors="ignore")
    return out
