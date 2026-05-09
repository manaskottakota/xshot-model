"""Primary (probabilistic) + secondary thresholded metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilistic(
    y_true: np.ndarray, y_prob: np.ndarray, *, model_name: str
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    out: dict[str, Any] = {"model": model_name}
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["log_loss"] = float(log_loss(y_true, y_prob))
    y_hat = (y_prob >= 0.5).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, y_hat))
    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    return out


def format_metric_block(rows: list[dict[str, Any]], *, title: str | None = None) -> str:
    lines = []
    if title:
        lines.append(title)
    for r in rows:
        prim = (
            f"ROC-AUC={r['roc_auc']:.4f}  log-loss={r['log_loss']:.4f}"
        )
        sec = (
            f"acc={r['accuracy']:.4f}  P={r['precision']:.4f}"
            f"  R={r['recall']:.4f}  F1={r['f1']:.4f}"
        )
        lines.append(f"{r['model']}: {prim} | secondary: {sec}")
    return "\n".join(lines)


def training_report_text(
    metrics_val: list[dict[str, Any]],
    metrics_test: list[dict[str, Any]],
    *,
    primary: str | None = None,
    calibrated_test: dict[str, Any] | None = None,
) -> str:
    parts = [
        format_metric_block(metrics_val, title="validation (model selection)"),
        "",
        format_metric_block(metrics_test, title="held-out test"),
    ]
    if primary:
        parts.extend(["", f"primary_model (by val log-loss): {primary}"])
    if calibrated_test:
        parts.extend(
            [
                "",
                (
                    f"calibrated primary on test — "
                    f"ROC-AUC={calibrated_test['roc_auc']:.4f}  "
                    f"log-loss={calibrated_test['log_loss']:.4f}"
                ),
            ]
        )
    return "\n".join(parts)
