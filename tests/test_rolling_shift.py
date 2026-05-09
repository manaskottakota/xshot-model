"""Regression tests for chronological rolling logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xshot.features.core import add_core_features
from xshot.features.rolling import add_rolling_player_features


def test_prior_cumulative_excludes_current_shot_make():
    base = pd.DataFrame(
        {
            "LOC_X": [0] * 3,
            "LOC_Y": [50] * 3,
            "SHOT_DISTANCE": [5] * 3,
            "SHOT_ZONE_BASIC": ["Restricted Area"] * 3,
            "SHOT_ZONE_AREA": ["Center(C)"] * 3,
            "SHOT_ZONE_RANGE": ["Less Than 8 ft."] * 3,
            "PERIOD": [1, 1, 1],
            "MINUTES_REMAINING": [10, 9, 8],
            "SECONDS_REMAINING": [0, 30, 0],
            "ACTION_TYPE": ["Layup"] * 3,
            "SHOT_TYPE": ["2PT Field Goal"] * 3,
            "SHOT_MADE_FLAG": [1, 1, 0],
            "PLAYER_ID": [7, 7, 7],
            "GAME_DATE": ["20231101"] * 3,
            "GAME_ID": ["g1", "g1", "g1"],
            "GAME_EVENT_ID": [10, 20, 30],
            "season": ["2023-24"] * 3,
            "shooting_team_is_home": [1] * 3,
            "score_diff_shooting_perspective": [0] * 3,
            "season_type": ["Regular Season"] * 3,
        }
    )
    core = add_core_features(base)
    rolled = add_rolling_player_features(core, last_n_games=2, global_prior=0.35)
    pcts = rolled["prior_cum_fg_pct"].to_numpy(dtype=float)
    np.testing.assert_almost_equal(pcts[0], 0.35, decimal=6)
    np.testing.assert_almost_equal(pcts[1], 1.0, decimal=6)
    np.testing.assert_almost_equal(pcts[2], 1.0, decimal=6)
