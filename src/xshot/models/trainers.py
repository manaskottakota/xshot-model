"""Estimator factories + preprocessing for tabular FG models."""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


GB_MODEL_NAMES = frozenset({"xgboost", "histogram_gradient_boosting"})


def _filter_init_kw(cls: type, kw: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "kwargs"}
    out = {k: v for k, v in kw.items() if k in allowed}
    # XGBoost **kwargs swallows unknowns in some versions — still filter for safety
    return out


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
        max_iter=2000,
        max_depth=12,
        max_leaf_nodes=64,
        learning_rate=0.04,
        l2_regularization=1e-2,
        min_samples_leaf=8,
        early_stopping=True,
        validation_fraction=0.04,
        n_iter_no_change=45,
        random_state=42,
    )

    if importlib.util.find_spec("xgboost") is not None:
        try:
            from xgboost import XGBClassifier  # type: ignore[import-untyped]

            xgb_kw: dict[str, Any] = dict(
                n_estimators=2000,
                max_depth=9,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.82,
                reg_lambda=1.8,
                min_child_weight=3,
                objective="binary:logistic",
                random_state=42,
                enable_categorical=False,
                verbosity=0,
                eval_metric="logloss",
            )
            if "early_stopping_rounds" in inspect.signature(XGBClassifier.__init__).parameters:
                xgb_kw["early_stopping_rounds"] = 65
            clf = XGBClassifier(**_filter_init_kw(XGBClassifier, xgb_kw))
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


def fit_classifier_pipeline(
    model_name: str,
    pipe: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: Any,
    X_va: pd.DataFrame,
    y_va: Any,
) -> Pipeline:
    """
    Train tabular pipelines.

    Tree boosters use validation rows for **XGBoost** early stopping; histogram
    boosting relies on sklearn's built-in holdout during ``fit``.
    """
    if model_name not in GB_MODEL_NAMES:
        pipe.fit(X_tr, y_tr)
        return pipe

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    x_tr = pre.fit_transform(X_tr, y_tr)
    x_va = pre.transform(X_va)
    if model_name == "xgboost":
        clf.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    else:
        clf.fit(x_tr, y_tr)
    return pipe
