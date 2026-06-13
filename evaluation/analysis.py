"""
Error analysis — digs into where and how a trained model fails.
All functions take arrays (predictions already computed) to stay pure
and testable without loading models or hitting the DB.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd


# ── confusion matrix analysis ─────────────────────────────────────────────────

def confusion_matrix_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    top_n_confused: int = 5,
) -> dict:
    """
    Deep analysis of where a classifier fails.

    Returns:
      - error_rate_per_class: which classes are hardest to predict
      - most_confused_pairs: which class pairs get mixed up most often
      - correct_count / error_count / total per class
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = class_names or [str(c) for c in sorted(np.unique(y_true))]

    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(classes)

    # per-class error rates
    per_class: dict[str, dict] = {}
    for i, cls in enumerate(classes):
        if i >= cm.shape[0]:
            continue
        total = int(cm[i].sum())
        correct = int(cm[i, i]) if i < cm.shape[1] else 0
        errors = total - correct
        per_class[cls] = {
            "total": total,
            "correct": correct,
            "errors": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
        }

    # most confused class pairs (off-diagonal, sorted by count)
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j or i >= cm.shape[0] or j >= cm.shape[1]:
                continue
            count = int(cm[i, j])
            if count > 0:
                confused_pairs.append({
                    "true_class": classes[i] if i < len(classes) else str(i),
                    "predicted_as": classes[j] if j < len(classes) else str(j),
                    "count": count,
                    "rate_of_true_class": round(count / max(per_class.get(classes[i], {}).get("total", 1), 1), 4),
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # summary
    total_samples = int(len(y_true))
    total_correct = int((y_true == y_pred).sum())
    hardest_class = max(per_class, key=lambda c: per_class[c]["error_rate"]) if per_class else None

    return {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_error_rate": round(1 - total_correct / total_samples, 4) if total_samples else 0.0,
        "error_rate_per_class": per_class,
        "most_confused_pairs": confused_pairs[:top_n_confused],
        "hardest_class": hardest_class,
        "confusion_matrix": cm.tolist(),
    }


# ── feature importance analysis ───────────────────────────────────────────────

def feature_importance_analysis(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
) -> dict:
    """
    Extract per-feature importance. Supports:
      - Tree models: feature_importances_ (mean decrease in impurity)
      - Linear models: abs(coef_) — magnitude of coefficient
      - Fallback: permutation importance (not implemented here — Phase 7+)

    Returns sorted list from most to least important.
    """
    importances: Optional[np.ndarray] = None
    method = "unknown"

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        method = "mean_decrease_impurity"

    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef[0] if coef.ndim == 2 else coef)
        method = "coefficient_magnitude"

    if importances is None:
        return {
            "method": "not_available",
            "note": "This model type does not expose feature importances directly. "
                    "Use permutation importance (Phase 7+).",
            "features": [],
        }

    if len(importances) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # normalise to [0, 1]
    total = importances.sum()
    normalised = importances / total if total > 0 else importances

    ranked = sorted(
        zip(feature_names, importances.tolist(), normalised.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    cumulative = 0.0
    features = []
    for name, raw, norm in ranked[:top_n]:
        cumulative += norm
        features.append({
            "feature": name,
            "importance": round(float(raw), 6),
            "importance_pct": round(float(norm * 100), 2),
            "cumulative_pct": round(float(cumulative * 100), 2),
        })

    # features needed to explain 80% and 95% of total importance
    threshold_features = {}
    for threshold in (0.80, 0.95):
        cum = 0.0
        for i, (_, _, norm) in enumerate(ranked):
            cum += norm
            if cum >= threshold:
                threshold_features[f"features_for_{int(threshold*100)}pct"] = i + 1
                break

    return {
        "method": method,
        "total_features": len(feature_names),
        "top_features": features,
        **threshold_features,
    }


# ── residual analysis (regression) ───────────────────────────────────────────

def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n_errors: int = 10,
) -> dict:
    """
    Analyse regression errors.

    Checks for:
      - Bias: mean residual ≠ 0 → model systematically over/under-predicts
      - Heteroscedasticity: residual variance grows with predicted value
      - Normality: skewness / kurtosis of residuals
      - Worst predictions: index positions of largest absolute errors
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    # bias
    mean_residual = float(residuals.mean())
    bias_direction = "over-predicts" if mean_residual < 0 else ("under-predicts" if mean_residual > 0 else "unbiased")

    # heteroscedasticity check: correlation between |residual| and predicted value
    corr = float(np.corrcoef(y_pred, abs_residuals)[0, 1])
    heteroscedastic = abs(corr) > 0.3  # rule of thumb threshold

    # normality indicators
    residual_series = pd.Series(residuals)
    skewness = round(float(residual_series.skew()), 4)
    kurtosis = round(float(residual_series.kurtosis()), 4)

    # worst predictions
    worst_indices = np.argsort(abs_residuals)[::-1][:top_n_errors].tolist()

    # percentile breakdown
    percentiles = [50, 75, 90, 95, 99]
    error_percentiles = {
        f"p{p}": round(float(np.percentile(abs_residuals, p)), 4)
        for p in percentiles
    }

    return {
        "bias": {
            "mean_residual": round(mean_residual, 4),
            "direction": bias_direction,
        },
        "variance": {
            "std_residual": round(float(residuals.std()), 4),
            "heteroscedastic": heteroscedastic,
            "pred_vs_residual_corr": round(corr, 4),
        },
        "normality": {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "approximately_normal": abs(skewness) < 1.0 and abs(kurtosis) < 3.0,
        },
        "error_percentiles": error_percentiles,
        "worst_prediction_indices": worst_indices,
    }


# ── error sample inspection ───────────────────────────────────────────────────

def error_examples(
    X: np.ndarray | pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 10,
    task: str = "classification",
) -> list[dict]:
    """
    Return the `n` worst predictions for manual inspection.
    Classification: misclassified samples.
    Regression: samples with largest absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "classification":
        error_mask = y_true != y_pred
        error_indices = np.where(error_mask)[0][:n]
    else:
        abs_err = np.abs(y_true - y_pred)
        error_indices = np.argsort(abs_err)[::-1][:n]

    examples = []
    for idx in error_indices:
        entry = {
            "index": int(idx),
            "y_true": float(y_true[idx]) if task == "regression" else str(y_true[idx]),
            "y_pred": float(y_pred[idx]) if task == "regression" else str(y_pred[idx]),
        }
        if task == "regression":
            entry["abs_error"] = round(abs(float(y_true[idx]) - float(y_pred[idx])), 4)
        if isinstance(X, pd.DataFrame):
            entry["features"] = {
                col: round(float(X.iloc[idx][col]), 4)
                if isinstance(X.iloc[idx][col], (int, float, np.floating))
                else str(X.iloc[idx][col])
                for col in X.columns
            }
        examples.append(entry)

    return examples
