"""
Model evaluation — computes classification and regression metrics from predictions.
All functions are pure: they take arrays, return dicts. No model or DB dependencies.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd


# ── classification ────────────────────────────────────────────────────────────

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Full classification report: accuracy, precision, recall, F1, ROC-AUC,
    per-class breakdown, and confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = class_names or [str(c) for c in sorted(np.unique(y_true))]
    n_classes = len(classes)
    avg = "binary" if n_classes == 2 else "weighted"

    metrics: dict[str, Any] = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }

    # ROC-AUC requires probability scores
    if y_proba is not None:
        try:
            if n_classes == 2:
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, proba)), 4)
            else:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 4
                )
        except ValueError:
            pass  # single class in eval set

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # per-class breakdown
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_support = cm.sum(axis=1).tolist()

    metrics["per_class"] = {
        cls: {
            "precision": round(float(per_class_p[i]), 4),
            "recall":    round(float(per_class_r[i]), 4),
            "f1":        round(float(per_class_f[i]), 4),
            "support":   int(per_class_support[i]),
        }
        for i, cls in enumerate(classes)
        if i < len(per_class_p)
    }

    # simple class imbalance flag
    supports = [v["support"] for v in metrics["per_class"].values()]
    if supports:
        imbalance_ratio = max(supports) / (min(supports) or 1)
        metrics["class_imbalance_ratio"] = round(imbalance_ratio, 2)

    return metrics


# ── regression ────────────────────────────────────────────────────────────────

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, R², MAPE, and residual summary statistics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    mape = None
    nonzero = y_true != 0
    if nonzero.any():
        mape = round(float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100), 4)

    return {
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": mape,
        "residuals": {
            "mean":   round(float(residuals.mean()), 4),
            "std":    round(float(residuals.std()), 4),
            "min":    round(float(residuals.min()), 4),
            "max":    round(float(residuals.max()), 4),
            "skew":   round(float(pd.Series(residuals).skew()), 4),
        },
    }


# ── unified entrypoint ────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "classification",
    class_names: Optional[list[str]] = None,
) -> dict:
    """Run the model on X_test and compute metrics for the given task."""
    y_pred = model.predict(X_test)

    if task == "classification":
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass
        return evaluate_classification(y_test, y_pred, y_proba, class_names)

    return evaluate_regression(y_test, y_pred)


# ── bootstrap confidence interval ─────────────────────────────────────────────

def bootstrap_confidence_interval(
    metric_value: float,
    n_samples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Analytical approximation of CI for a proportion (e.g. accuracy).
    Uses the Wilson score interval — more accurate than normal approx for
    extreme proportions and small n.
    """
    from scipy.stats import norm

    z = norm.ppf((1 + confidence) / 2)
    p = metric_value
    n = n_samples
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    lower = round(max(0.0, centre - half_width), 4)
    upper = round(min(1.0, centre + half_width), 4)
    return lower, upper
