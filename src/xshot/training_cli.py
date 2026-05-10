"""End-to-end dataset build, trains three estimators, writes metrics + artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from xshot import __version__
from xshot.datasets import (
    ROLLING_LAST_N_GAMES,
    X_y_from_table,
    build_feature_table,
    feature_columns,
    load_or_fetch_shots_cached,
    merge_all_context,
    resolve_ablation_union,
    training_manifest_row,
)
from xshot.feature_analysis import build_feature_diagnostic_bundle
from xshot.metrics import evaluate_probabilistic, training_report_text
from xshot.models.trainers import build_models, fit_classifier_pipeline
from xshot.run_tracking import (
    append_training_run_record,
    format_previous_run_compare,
    read_last_completed_run,
)
from xshot.splits import masks_from_seasons


def _parse_seasons(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_teams(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_fields(s: str | None) -> frozenset[str]:
    if not s:
        return frozenset()
    return frozenset(x.strip() for x in s.split(",") if x.strip())


def _sanitize_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in s)[:140]


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
    omit_features: frozenset[str] | None = None,
    ablation_variant: str | None = None,
    feature_diagnostics: bool = False,
    diagnostics_perm_repeats: int = 6,
    diagnostics_max_perm_rows: int | None = 80_000,
    diagnostics_seed: int = 42,
    metrics_history_path: Path | None = None,
) -> list[dict]:
    omit_eff = sorted(omit_features or frozenset())
    omit_frozen = frozenset(omit_eff)
    _, cat = feature_columns(features, omit_features=omit_frozen)
    X, y = X_y_from_table(table, features=features, omit_features=omit_frozen)

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
    valid_metrics: list[dict] = []

    for name, est in models_cfg.items():
        fit_classifier_pipeline(name, est, X_tr, y_tr, X_va, y_va)
        prob_va = est.predict_proba(X_va)[:, 1]
        row = evaluate_probabilistic(y_va, prob_va, model_name=name)
        row["split"] = "validation"
        valid_metrics.append(row)

    for name, est in models_cfg.items():
        prob_te = est.predict_proba(X_te)[:, 1]
        fitted[name] = est
        results.append(evaluate_probabilistic(y_te, prob_te, model_name=name))

    for name, est in fitted.items():
        joblib.dump(est, out_dir / f"{name}.joblib")

    primary = min(valid_metrics, key=lambda r: r["log_loss"])["model"]
    primary_est = fitted[primary]
    joblib.dump(primary_est, out_dir / "xshot_primary.joblib")

    cal_test_row: dict | None = None
    cal: CalibratedClassifierCV
    try:
        cal = CalibratedClassifierCV(estimator=primary_est, method="isotonic", cv=3)
        cal.fit(X_va, y_va)
    except Exception:
        cal = CalibratedClassifierCV(estimator=primary_est, method="sigmoid", cv=3)
        cal.fit(X_va, y_va)

    joblib.dump(cal, out_dir / "xshot_primary_calibrated.joblib")
    prob_te_cal = cal.predict_proba(X_te)[:, 1]
    cal_test_row = evaluate_probabilistic(y_te, prob_te_cal, model_name=f"{primary}_calibrated")

    diagnostics_path_written: Path | None = None
    if feature_diagnostics and cal_test_row is not None:
        bundle = build_feature_diagnostic_bundle(
            X_tr,
            y_tr,
            categorical=list(cat),
            fitted_primary_model=primary_est,
            n_perm_repeats=diagnostics_perm_repeats,
            rng_seed=int(diagnostics_seed),
            max_rows_perm=diagnostics_max_perm_rows,
        )
        diagnostics_path_written = out_dir / "feature_diagnostics.json"
        diagnostics_path_written.write_text(json.dumps(bundle, indent=2))

    manifest = training_manifest_row(
        seasons=sorted(set(table["season"].astype(str))),
        n_rows=len(table),
        features=features,
        pbp_max_games=pbp_max_games,
        tracking_path=str(tracking_path) if tracking_path else None,
    )

    summary = {
        "xshot_version": __version__,
        "rolling_last_n_games": ROLLING_LAST_N_GAMES,
        "omit_features": omit_eff,
        "feature_ablation_tag": ablation_variant or "",
        "manifest": manifest,
        "split": {"train": train_seasons, "val": val_seasons, "test": test_seasons},
        "metrics_test": results,
        "metrics_val": valid_metrics,
        "primary_model": primary,
        "metrics_test_calibrated_primary": cal_test_row,
    }
    if diagnostics_path_written is not None:
        summary["feature_diagnostics_path"] = str(diagnostics_path_written)

    previous_run: dict | None = None
    if metrics_history_path is not None:
        previous_run = read_last_completed_run(metrics_history_path)

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    human_base = training_report_text(
        valid_metrics,
        results,
        primary=primary,
        calibrated_test=cal_test_row,
    )
    compare_block = format_previous_run_compare(
        {"calibrated_primary_test": dict(cal_test_row)},
        previous_run,
    )
    human_full = human_base + "\n\n" + compare_block
    (out_dir / "metrics_human.txt").write_text(human_full)

    print(human_full)

    if metrics_history_path is not None and cal_test_row is not None:
        append_training_run_record(
            metrics_history_path,
            out_dir=out_dir,
            manifest=dict(manifest),
            calibrated_test_row=dict(cal_test_row),
            primary_model=str(primary),
            omit_features=list(omit_eff),
            ablation_variant=ablation_variant,
            feature_diagnostics_path=(
                str(diagnostics_path_written) if diagnostics_path_written else None
            ),
        )

    print("\n[Wrote artifacts to]", str(out_dir))
    return results


def _ablation_run_specs(groups_csv: str | None, *, each_group: bool) -> Sequence[tuple[str, frozenset[str]]]:
    if not groups_csv:
        return (("", frozenset()),)
    names = [g.strip() for g in groups_csv.split(",") if g.strip()]
    if not names:
        return (("", frozenset()),)
    if each_group:
        specs: list[tuple[str, frozenset[str]]] = [("baseline", frozenset())]
        for n in names:
            specs.append((n, resolve_ablation_union([n])))
        return specs
    tag = "|".join(names)
    union = resolve_ablation_union(names)
    return ((tag, union),)


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
        "--omit-features",
        default=None,
        help="Comma-separated raw feature tokens to withhold from ``X`` (reversible omission).",
    )
    parser.add_argument(
        "--ablate-groups",
        default=None,
        help=(
            "Comma-separated predefined group keys (see FEATURE_ABLATION_GROUPS in datasets) "
            "to omit jointly from ``X``. With --ablate-each-group run baseline plus one run "
            "per listed group separately."
        ),
    )
    parser.add_argument(
        "--ablate-each-group",
        action="store_true",
        help="When combined with --ablate-groups iterate baseline + isolated group drops.",
    )
    parser.add_argument(
        "--feature-diagnostics",
        action="store_true",
        help="Write permutation / missingness summaries to ``feature_diagnostics.json``.",
    )
    parser.add_argument(
        "--diag-perm-repeats",
        type=int,
        default=6,
        help="Repeated shuffles inside sklearn permutation_importance.",
    )
    parser.add_argument(
        "--diag-max-perm-rows",
        type=int,
        default=80_000,
        help="Subsample rows for permutation importance (omit or zero for full train rows).",
    )
    parser.add_argument(
        "--diag-seed",
        type=int,
        default=42,
        help="Random seed controlling diagnostics subsampling + permutation repeats.",
    )
    parser.add_argument(
        "--metrics-history",
        type=Path,
        default=Path("artifacts/metrics_history.jsonl"),
        help=(
            "Append run metadata + calibrated-test KPIs here (newline-delimited JSON); "
            "use --no-metrics-history to skip."
        ),
    )
    parser.add_argument(
        "--no-metrics-history",
        action="store_true",
        help="Disable JSONL incremental history writes + prior-run comparison footing.",
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
        default=None,
        help="Random subsample cap (omit or <=0 for full data after season filtering).",
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
    max_rows = (
        None if args.max_rows is None or args.max_rows <= 0 else args.max_rows
    )

    history_path = None if args.no_metrics_history else args.metrics_history
    user_omit = _parse_csv_fields(args.omit_features)
    ablation_specs = _ablation_run_specs(args.ablate_groups, each_group=args.ablate_each_group)

    feature_modes = (
        ["core", "core+advanced"] if args.ablation else [args.features]
    )
    diag_max = None if args.diag_max_perm_rows <= 0 else int(args.diag_max_perm_rows)

    for feat in feature_modes:
        for ab_tag, group_omit in ablation_specs:
            eff_omit = frozenset(user_omit | group_omit)

            slug_parts = []
            if len(feature_modes) > 1:
                slug_parts.append(_sanitize_slug(feat.replace("+", "_")))
            label = ""
            if ab_tag:
                label = _sanitize_slug(ab_tag)
                slug_parts.append(label if label != "baseline" else "")
            slug_parts = [p for p in slug_parts if p]
            slug = "_".join(slug_parts)

            out = (
                args.out_dir
                if len(feature_modes) == 1 and not slug
                else args.out_dir.with_name(args.out_dir.name + (("_" + slug) if slug else ""))
            )

            variant_name = ab_tag if ab_tag else None
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
                omit_features=eff_omit,
                ablation_variant=variant_name,
                feature_diagnostics=bool(args.feature_diagnostics),
                diagnostics_perm_repeats=int(args.diag_perm_repeats),
                diagnostics_max_perm_rows=diag_max,
                diagnostics_seed=args.diag_seed,
                metrics_history_path=history_path,
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
