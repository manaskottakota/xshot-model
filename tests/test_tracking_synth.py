"""Deterministic surrogate tracking geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xshot.features.tracking_synth import (
    contest_bucket_ix,
    defender_xy_inches_batch,
    heuristic_def_distance_ft,
    stable_jitter01,
)


def test_stable_hash_is_deterministic():
    df = pd.DataFrame({"GAME_ID": ["a", "a"], "GAME_EVENT_ID": ["1", "2"]})
    u1 = stable_jitter01(df, salt=1)
    u2 = stable_jitter01(df, salt=1)
    assert u1.shape == u2.shape == (2,)
    assert (u1 == u2).all()


def test_def_distance_heuristic_moves_with_shot_depth():
    u = np.linspace(0, 1, 5).astype(np.float64)
    d_close = heuristic_def_distance_ft(
        shot_distance_ft=np.full(5, 6.0),
        abs_loc_x_ft=np.zeros(5),
        u01=u,
    )
    d_far = heuristic_def_distance_ft(
        shot_distance_ft=np.full(5, 24.0),
        abs_loc_x_ft=np.zeros(5),
        u01=u,
    )
    assert (d_far >= d_close - 1e-6).all()


def test_contest_bucket_monotone():
    pts = contest_bucket_ix([1.9, 3.9, 5.9, 8.9])
    assert list(map(int, pts)) == [0, 1, 2, 3]


def test_defender_spacing_slider_motion():
    lx = np.asarray([-14.5, -14.5], dtype=float)
    ly = np.asarray([13.25, 13.25], dtype=float)
    d1 = np.asarray([3.05, 7.92], dtype=float)
    psi = np.asarray([-6.25, -6.25], dtype=float)
    x_in, y_in = defender_xy_inches_batch(lx, ly, d1, psi)
    drift = np.hypot(x_in[0] - x_in[1], y_in[0] - y_in[1])
    assert drift >= 56.25
