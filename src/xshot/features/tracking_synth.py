"""Synthetic / fallback SportVU proxies when telemetry merge is missing or partial.

Training without a SportVU parquet would otherwise leave defender / touch /
movement scalars entirely NaN, so downstream median-imputation wipes per-row signal.
These helpers synthesize plausible, smoothly varying substitutes keyed by GAME_ID /
GAME_EVENT_ID so tree models can exploit advanced columns.

Inference LAB rows pass explicit user fields; callers may still reuse the same
XYZ placement math for coherence with training geometry.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def stable_jitter01(frame: pd.DataFrame, *, salt: int = 0xCAFEBABECAFEBABF) -> np.ndarray:
    """
    Stable [0,1) pseudo-uniform per row using pandas hash — no RNG drift across runs.

    Columns should identify a shot uniquely (typically GAME_ID + GAME_EVENT_ID).
    """
    cols = frame.astype(str).fillna("")
    h = pd.util.hash_pandas_object(cols).to_numpy(dtype=np.uint64)
    salted = np.bitwise_xor(h, np.uint64(salt))
    denom = np.float64(np.uint64(1 << 61))
    return (salted.astype(np.float64) % denom) / denom


def heuristic_def_distance_ft(
    shot_distance_ft: np.ndarray | pd.Series,
    abs_loc_x_ft: np.ndarray | pd.Series,
    u01: np.ndarray,
    *,
    min_ft: float = 2.75,
    max_ft: float = 9.6,
) -> np.ndarray:
    """Scene-dependent close-def heuristic + keyed jitter."""
    sd = pd.to_numeric(pd.Series(np.asarray(shot_distance_ft)), errors="coerce").to_numpy(
        dtype=np.float64
    )
    ax = pd.to_numeric(pd.Series(np.asarray(abs_loc_x_ft)), errors="coerce").fillna(
        0.0
    ).to_numpy(dtype=np.float64)
    u = np.asarray(u01, dtype=np.float64)
    base = 2.88 + 0.062 * sd - 0.017 * ax + (u - 0.5) * 1.06
    return np.clip(np.where(np.isfinite(base), base, np.nan), min_ft, max_ft)


def contest_azimuth_from_u(u01_secondary: np.ndarray, *, span_deg: float = 76.5) -> np.ndarray:
    return (np.asarray(u01_secondary, dtype=np.float64) * 2.0 - 1.0) * span_deg


def defender_xy_inches_batch(
    loc_x_ft: np.ndarray | pd.Series,
    loc_y_ft: np.ndarray | pd.Series,
    defender_distance_ft: np.ndarray | pd.Series,
    contest_azimuth_deg: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector planar placement: defender sits ``def_ft`` from shooter toward rim ± azimuth."""

    lx = pd.to_numeric(pd.Series(np.asarray(loc_x_ft)), errors="coerce").fillna(
        0.0
    ).to_numpy(dtype=np.float64)
    ly = pd.to_numeric(pd.Series(np.asarray(loc_y_ft)), errors="coerce").fillna(
        0.0
    ).to_numpy(dtype=np.float64)
    dd = pd.to_numeric(
        pd.Series(np.asarray(defender_distance_ft)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    psi_deg = pd.to_numeric(
        pd.Series(np.asarray(contest_azimuth_deg)),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=np.float64)
    psi = np.radians(psi_deg)

    s = np.column_stack([lx * 12.0, ly * 12.0])
    toward_rim = np.column_stack([-lx * 12.0, -ly * 12.0])
    n = np.linalg.norm(toward_rim, axis=1, keepdims=True)
    n[n < 1e-9] = 1e-9
    u = toward_rim / n

    # Perpendicular clockwise (NBA chart orientation doesn't need exact chirality —
    # we only require smooth variation keyed by ψ).
    perp = np.column_stack([-u[:, 1], u[:, 0]])

    psi_f = psi.astype(np.float64, copy=False)
    cos = np.cos(psi_f)[:, np.newaxis]
    sin = np.sin(psi_f)[:, np.newaxis]
    ray = cos * u + sin * perp

    rn = np.linalg.norm(ray, axis=1, keepdims=True)
    rn[rn < 1e-9] = 1e-9
    ray_n = ray / rn

    d_in = np.clip(dd, 0.09, np.inf) * 12.0
    xy = s + ray_n * d_in[:, None]
    return xy[:, 0], xy[:, 1]


def contest_bucket_ix(distance_ft: np.ndarray | pd.Series) -> np.ndarray:
    """Integer bucket 0=smother … 3=wide open aligned with sklearn-friendly floats later."""
    d = pd.to_numeric(pd.Series(np.asarray(distance_ft)), errors="coerce").to_numpy(
        dtype=np.float64
    )
    out = np.full(d.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(d)
    # Mirrors prior pd.cut bins=(-∞, 2.5], (2.5,4.5], (4.5,6.5], (6.5, ∞)
    b0 = ok & (d <= 2.5)
    b1 = ok & (~b0 & (d <= 4.5))
    b2 = ok & (~(b0 | b1) & (d <= 6.5))
    b3 = ok & (~(b0 | b1 | b2))
    out[b0] = 0.0
    out[b1] = 1.0
    out[b2] = 2.0
    out[b3] = 3.0
    return out


def defender_rel_angle_rad(dx_in: np.ndarray, dy_in: np.ndarray, shx_in: np.ndarray, shy_in: np.ndarray) -> np.ndarray:
    vx = dx_in.astype(np.float64) - shx_in.astype(np.float64)
    vy = dy_in.astype(np.float64) - shy_in.astype(np.float64)
    return np.arctan2(vx, np.maximum(np.abs(vy), 1e-3))


def synth_scalar_def_xy_inches(lx_ft: float, ly_ft: float, d_ft: float, contest_deg: float) -> tuple[float, float]:
    ux, uy = defender_xy_inches_batch(
        np.array([lx_ft], dtype=float),
        np.array([ly_ft], dtype=float),
        np.array([d_ft], dtype=float),
        np.array([contest_deg], dtype=float),
    )
    return float(ux[0]), float(uy[0])
