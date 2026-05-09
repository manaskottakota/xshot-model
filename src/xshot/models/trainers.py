"""Estimator factories + preprocessing for tabular FG models."""

from __future__ import annotations

import importlib.util
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_cat_split(
    X: pd.DataFrame, categorical: list[str]
) -> tuple[list[str], list[str]]:
    cat = [c for c in categorical if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def build_preprocessor(
    X_fit: pd.DataFrame, categorical: list[str]
) -> ColumnTransformer:
    num_cols, cat_cols = numeric_cat_split(X_fit, categorical)
    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            (
                "oh",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_lr(cat_cols: list[str], X_fit: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_fit, cat_cols)
    clf = LogisticRegression(
        max_iter=800,
        class_weight=None,
        C=2.5,
        solver="lbfgs",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def build_rf(cat_cols: list[str], X_fit: pd.DataFrame) -> Pipeline:
    """RandomForest ignores scaling but shares imputation/OHE."""

    num_cols, cats = numeric_cat_split(X_fit, cat_cols)

    numeric_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cats),
        ],
        remainder="drop",
    )
    rf = RandomForestClassifier(
        n_estimators=160,
        max_depth=22,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", ct), ("clf", rf)])


def _rf_pre(cat_cols: list[str], X_fit: pd.DataFrame) -> ColumnTransformer:
    return build_rf(cat_cols, X_fit).named_steps["pre"]


def build_gradient_boost(cat_cols: list[str], X_fit: pd.DataFrame) -> tuple[str, Pipeline]:
    """
    Prefer **XGBoost** when installed and the native runtime loads cleanly;
    otherwise fall back to sklearn ``HistGradientBoostingClassifier`` (same sklearn
    ``Pipeline(pre → clf)`` contract).
    """
    rf_pre = _rf_pre(cat_cols, X_fit)
    gb_name = "histogram_gradient_boosting"
    clf: Any = HistGradientBoostingClassifier(
        max_iter=520,
        max_depth=12,
        max_leaf_nodes=64,
        learning_rate=0.05,
        l2_regularization=1e-2,
        min_samples_leaf=8,
        random_state=42,
    )

    if importlib.util.find_spec("xgboost") is not None:
        try:
            from xgboost import XGBClassifier  # type: ignore[import-untyped]

            clf = XGBClassifier(
                n_estimators=380,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.8,
                min_child_weight=3,
                objective="binary:logistic",
                random_state=42,
                enable_categorical=False,
                verbosity=0,
            )
            gb_name = "xgboost"
        except Exception:  # pragma: no cover - dlopen / OMP issues
            gb_name = "histogram_gradient_boosting"

    return gb_name, Pipeline([("pre", rf_pre), ("clf", clf)])


def build_models(X_fit: pd.DataFrame, categorical: list[str], *, rng: Any = None) -> dict[str, Pipeline]:
    _ = rng
    gb_label, gb_pipe = build_gradient_boost(categorical, X_fit)
    return {
        "logistic_regression": build_lr(categorical, X_fit),
        "random_forest": build_rf(categorical, X_fit),
        gb_label: gb_pipe,
    }


def estimate_feature_matrix(model: Pipeline, X: pd.DataFrame) -> Any:
    return model.named_steps["pre"].transform(X)
