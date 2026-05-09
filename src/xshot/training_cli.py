"""End-to-end dataset build, trains three estimators, writes metrics + artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from xshot import __version__
from xshot.datasets import (
    ROLLING_LAST_N_GAMES,
    X_y_from_table,
    build_feature_table,
    feature_columns,
    load_or_fetch_shots_cached,
    merge_all_context,
    training_manifest_row,
)
from xshot.metrics import evaluate_probabilistic, format_metric_block
from xshot.models.trainers import build_models
from xshot.splits import masks_from_seasons


def _parse_seasons(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_teams(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_full_table(
    seasons: list[str],
    *,
    features: str,
    team_ids: list[int] | None,
    pbp_max_games: int | None,
    tracking_path: Path | None,
    max_rows: int | None,
    seed: int,
    allowed_seasons: set[str],
) -> pd.DataFrame:
    raw_parts: list[pd.DataFrame] = []
    for season in seasons:
        cache = Path(f"data/raw/shots_{season.replace('/', '-')}_REG.parquet")
        rf = load_or_fetch_shots_cached(
            season,
            season_type="Regular Season",
            cache_shots=cache,
            team_ids=team_ids,
        )
        raw_parts.append(rf)

    shots = pd.concat(raw_parts, ignore_index=True)
    shots = shots[shots["season"].astype(str).isin(allowed_seasons)]

    if max_rows is not None and len(shots) > max_rows:
        shots = shots.sample(n=max_rows, random_state=seed)

    merged = merge_all_context(
        shots,
        tracking_path=tracking_path,
        pbp_max_games=pbp_max_games,
    )
    merged.attrs["pbp_max_games"] = pbp_max_games
    merged.attrs["tracking_path"] = str(tracking_path) if tracking_path else None
    return build_feature_table(merged, features=features)


def train_and_eval(
    table: pd.DataFrame,
    *,
    train_seasons: list[str],
    val_seasons: list[str],
    test_seasons: list[str],
    features: str,
    out_dir: Path,
    pbp_max_games: int | None,
    tracking_path: Path | None,
) -> list[dict]:
    _, cat = feature_columns(features)
    X, y = X_y_from_table(table, features=features)

    masks = masks_from_seasons(
        table["season"],
        train=set(train_seasons),
        val=set(val_seasons),
        test=set(test_seasons),
    )

    y_tr = y[masks.train.to_numpy(dtype=bool)]
    y_va = y[masks.val.to_numpy(dtype=bool)]
    y_te = y[masks.test.to_numpy(dtype=bool)]
    X_tr = X.loc[masks.train]
    X_va = X.loc[masks.val]
    X_te = X.loc[masks.test]

    out_dir.mkdir(parents=True, exist_ok=True)
    models_cfg = build_models(X_tr, cat)
    results: list[dict] = []
    fitted: dict[str, object] = {}
    for name, est in models_cfg.items():
        est.fit(X_tr, y_tr)
        prob_te = est.predict_proba(X_te)[:, 1]
        fitted[name] = est
        results.append(evaluate_probabilistic(y_te, prob_te, model_name=name))

    for name, est in fitted.items():
        joblib.dump(est, out_dir / f"{name}.joblib")

    valid_metrics: list[dict] = []
    for name, est in fitted.items():
        prob_va = est.predict_proba(X_va)[:, 1]
        row = evaluate_probabilistic(y_va, prob_va, model_name=name)
        row["split"] = "validation"
        valid_metrics.append(row)

    summary = {
        "xshot_version": __version__,
        "rolling_last_n_games": ROLLING_LAST_N_GAMES,
        "manifest": training_manifest_row(
            seasons=sorted(set(table["season"].astype(str))),
            n_rows=len(table),
            features=features,
            pbp_max_games=pbp_max_games,
            tracking_path=str(tracking_path) if tracking_path else None,
        ),
        "split": {"train": train_seasons, "val": val_seasons, "test": test_seasons},
        "metrics_test": results,
        "metrics_val": valid_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "metrics_human.txt").write_text(format_metric_block(results))

    print(format_metric_block(results))
    print("\n[Wrote artifacts to]", str(out_dir))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xshot-train")
    parser.add_argument(
        "--seasons",
        default="2021-22,2022-23,2023-24",
        help="Comma-separated seasons to download/cache before filtering to split seasons.",
    )
    parser.add_argument(
        "--train-seasons",
        default="2021-22",
        help="Comma-separated seasons for the training slice (disjoint from val/test).",
    )
    parser.add_argument(
        "--val-season",
        default="2022-23",
        help="Single season label for validation metrics / early diagnostics.",
    )
    parser.add_argument(
        "--test-season",
        default="2023-24",
        help="Single season held out for reported test metrics.",
    )
    parser.add_argument(
        "--features",
        choices=["core", "core+advanced"],
        default="core",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Train both core and core+advanced, writing sibling output folders.",
    )
    parser.add_argument(
        "--team-ids",
        default=None,
        help="Comma-separated TEAM_ID filters (limits HTTP volume).",
    )
    parser.add_argument(
        "--pbp-max-games",
        type=int,
        default=None,
        help="Cap unique games when downloading play-by-play (smoke tests).",
    )
    parser.add_argument(
        "--tracking-path",
        type=Path,
        default=None,
        help="Optional Parquet/CSV with tracking columns (see xshot.ingest.tracking).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=40_000,
        help="Random subsample cap after season filtering (None for all rows).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/run_default"),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    seasons_dl = _parse_seasons(args.seasons)
    train_ss = _parse_seasons(args.train_seasons)
    val_ss = _parse_seasons(args.val_season)
    test_ss = _parse_seasons(args.test_season)
    if len(val_ss) != 1 or len(test_ss) != 1:
        parser.error("--val-season and --test-season must each be a single season label")
    val_set = {val_ss[0]}
    test_set = {test_ss[0]}
    allowed = set(train_ss) | val_set | test_set
    for s in allowed:
        if s not in seasons_dl:
            parser.error(
                f"Season {s!r} used in splits but missing from --seasons download list"
            )

    team_ids = _parse_teams(args.team_ids)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    feature_modes = (
        ["core", "core+advanced"] if args.ablation else [args.features]
    )

    first = True
    for feat in feature_modes:
        out = args.out_dir if len(feature_modes) == 1 else args.out_dir.parent / (
            f"{args.out_dir.name}_{feat.replace('+', '_')}"
        )
        table = build_full_table(
            seasons_dl,
            features=feat,
            team_ids=team_ids,
            pbp_max_games=args.pbp_max_games,
            tracking_path=args.tracking_path,
            max_rows=max_rows,
            seed=args.seed,
            allowed_seasons=allowed,
        )
        train_and_eval(
            table,
            train_seasons=train_ss,
            val_seasons=list(val_set),
            test_seasons=list(test_set),
            features=feat,
            out_dir=out,
            pbp_max_games=args.pbp_max_games,
            tracking_path=args.tracking_path,
        )

    return 0
