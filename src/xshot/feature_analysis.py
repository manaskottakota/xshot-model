"""Feature diagnostics on labeled splits (missingness, correlations, permutation gain)."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def _numeric_frame(X: pd.DataFrame, categorical: list[str]) -> tuple[pd.DataFrame, list[str]]:
    cat_set = frozenset(categorical)
    num_cols = [c for c in X.columns if c not in cat_set]
    return X[num_cols].apply(pd.to_numeric, errors="coerce"), num_cols


def missing_value_fractions(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in df.columns:
        numer = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        bad = ~np.isfinite(numer)
        out[c] = float(np.mean(bad))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def pearson_with_binary_target(vec: pd.Series | np.ndarray, y_true: np.ndarray) -> float:
    v = pd.to_numeric(pd.Series(vec), errors="coerce").to_numpy(dtype=float)
    y = np.asarray(y_true, dtype=float)
    m = np.isfinite(v)
    if int(m.sum()) < 10:
        return float("nan")
    vn = v[m]
    yn = y[m]
    if np.std(vn) < 1e-12 or np.std(yn) < 1e-12:
        return float("nan")
    return float(np.corrcoef(vn, yn)[0, 1])


def correlations_numeric_with_target(num_df: pd.DataFrame, y_true: np.ndarray) -> dict[str, float]:
    out = {c: pearson_with_binary_target(num_df[c], y_true) for c in num_df.columns}
    return dict(sorted(out.items(), key=lambda kv: np.nan_to_num(abs(kv[1]), nan=0.0), reverse=True))


def high_numeric_correlation_pairs(df: pd.DataFrame, *, min_abs_corr: float) -> list[dict[str, Any]]:
    pair_df = pd.to_numeric(df, errors="coerce")
    valid = pair_df.dropna(axis=1, thresh=max(50, len(pair_df) // 40))
    c = valid.corr(method="pearson", numeric_only=True)
    pairs: list[dict[str, Any]] = []
    for a, b in combinations(c.columns, 2):
        try:
            r = float(c.loc[a, b])
        except Exception:
            continue
        if not np.isfinite(r):
            continue
        if abs(r) >= min_abs_corr:
            pairs.append({"feature_a": a, "feature_b": b, "pearson_r": round(r, 5)})
    pairs.sort(key=lambda d: abs(d["pearson_r"]), reverse=True)
    return pairs


def _subset_for_perm(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    rng: Any,
    max_rows: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y
    rnd = rng if hasattr(rng, "choice") else np.random.RandomState(int(rng) if rng is not None else 42)
    ix = rnd.choice(np.arange(len(X)), size=min(max_rows, len(X)), replace=False)
    return X.iloc[ix], y[ix]


def _transformed_export_names(primary: Pipeline) -> np.ndarray:
    pre = primary.named_steps.get("pre")
    if hasattr(pre, "get_feature_names_out"):
        names = np.asarray(pre.get_feature_names_out())
        if names.ndim == 1:
            return names.flat
        return names.flatten()
    n = getattr(getattr(primary, "named_steps", {}).get("clf", None), "n_features_in_", None)
    if n is None:
        return np.asarray([])
    return np.asarray([f"feature_{i}" for i in range(int(n))])


def build_feature_diagnostic_bundle(
    X: pd.DataFrame,
    y_true: np.ndarray,
    *,
    categorical: list[str],
    fitted_primary_model: Pipeline,
    n_perm_repeats: int,
    rng_seed: int,
    max_rows_perm: int | None,
) -> dict[str, Any]:
    num_df, _ = _numeric_frame(X, categorical)

    bundle: dict[str, Any] = {
        "n_rows_reference": len(X),
        "missing_frac_by_numeric_feature": missing_value_fractions(num_df),
        "pearson_with_target_numeric_sorted": correlations_numeric_with_target(num_df, y_true),
        "numeric_pairs_abs_pearson_ge_092": high_numeric_correlation_pairs(num_df, min_abs_corr=0.92)[:120],
        "permutation_importance_sorted": [],
    }

    Xp, yp = _subset_for_perm(X, np.asarray(y_true).astype(int), rng=rng_seed, max_rows=max_rows_perm)
    bundle["n_rows_perm"] = len(Xp)

    if n_perm_repeats <= 0:
        bundle["permutation_note"] = "skipped (n_perm_repeats<=0)"
        return bundle

    try:
        pi = permutation_importance(
            fitted_primary_model,
            Xp,
            yp,
            scoring="roc_auc",
            n_repeats=n_perm_repeats,
            random_state=int(rng_seed),
            n_jobs=-1,
        )
    except Exception as e:
        bundle["permutation_primary_error"] = repr(e)
        return bundle

    names = _transformed_export_names(fitted_primary_model)
    ranked: list[dict[str, Any]] = []
    for ix in range(len(pi.importances_mean)):
        lbl = names[ix] if ix < len(names) else f"f{ix}"
        ranked.append(
            {
                "transformed_component": str(lbl),
                "mean_delta_roc_auc": round(float(pi.importances_mean[ix]), 6),
                "std": round(float(pi.importances_std[ix]), 6),
            }
        )
    ranked.sort(key=lambda r: abs(r["mean_delta_roc_auc"]), reverse=True)

    bundle["permutation_importance_sorted"] = ranked[:100]
    return bundle
