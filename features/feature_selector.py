"""
Feature selection methods. All functions return a structured result dict so
callers can inspect scores alongside the selected column list.
"""

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

ScoringMethod = Literal["correlation", "spearman", "mutual_info"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _numeric_features(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in (exclude or [])]


# ── 1. Variance threshold ─────────────────────────────────────────────────────

def variance_threshold(
    df: pd.DataFrame,
    threshold: float = 0.01,
    exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Remove features whose variance falls below `threshold`.
    Near-zero variance features carry almost no signal.
    """
    candidates = _numeric_features(df, exclude)
    variances = df[candidates].var()

    selected = variances[variances >= threshold].index.tolist()
    dropped = variances[variances < threshold].index.tolist()

    return {
        "method": "variance_threshold",
        "threshold": threshold,
        "selected": selected,
        "dropped": dropped,
        "variances": variances.round(6).to_dict(),
    }


# ── 2. Statistical selection (correlation / mutual info) ──────────────────────

def statistical_selection(
    df: pd.DataFrame,
    target: str,
    method: ScoringMethod = "correlation",
    threshold: float = 0.05,
    exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Keep features whose absolute correlation (or mutual info) with `target`
    exceeds `threshold`. Low-correlation features add noise without signal.
    """
    candidates = _numeric_features(df, exclude=[target] + (exclude or []))
    target_series = df[target]

    scores: dict[str, float] = {}
    for col in candidates:
        col_series = df[col].dropna()
        common = col_series.index.intersection(target_series.dropna().index)
        if len(common) < 2:
            scores[col] = 0.0
            continue

        if method == "correlation":
            scores[col] = abs(col_series[common].corr(target_series[common]))
        elif method == "spearman":
            scores[col] = abs(col_series[common].corr(target_series[common], method="spearman"))
        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(
                col_series[common].values.reshape(-1, 1),
                target_series[common].values,
                random_state=42,
            )
            scores[col] = float(mi[0])

    scores = {k: round(v, 6) for k, v in scores.items()}
    selected = [c for c, s in scores.items() if s >= threshold]
    dropped = [c for c, s in scores.items() if s < threshold]

    return {
        "method": method,
        "threshold": threshold,
        "selected": sorted(selected, key=lambda c: scores[c], reverse=True),
        "dropped": dropped,
        "scores": dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
    }


# ── 3. Forward selection ──────────────────────────────────────────────────────

def forward_selection(
    df: pd.DataFrame,
    target: str,
    max_features: int = 20,
    cv: int = 3,
    exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Greedy forward selection: start with no features, iteratively add the one
    that most improves cross-validated R² (regression) until `max_features`.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    candidates = _numeric_features(df, exclude=[target] + (exclude or []))
    X_all = df[candidates].fillna(df[candidates].mean())
    y = df[target].fillna(df[target].mean())

    selected: list[str] = []
    remaining = candidates.copy()
    history: list[dict] = []
    best_score = -np.inf

    for _ in range(min(max_features, len(remaining))):
        step_scores: dict[str, float] = {}
        for col in remaining:
            cols = selected + [col]
            folds = min(cv, len(X_all))
            score = cross_val_score(
                LinearRegression(), X_all[cols], y, cv=folds, scoring="r2"
            ).mean()
            step_scores[col] = round(float(score), 6)

        best_col = max(step_scores, key=step_scores.get)
        if step_scores[best_col] <= best_score:
            break  # no improvement — stop early

        best_score = step_scores[best_col]
        selected.append(best_col)
        remaining.remove(best_col)
        history.append({"added": best_col, "r2": best_score})

    return {
        "method": "forward_selection",
        "selected": selected,
        "final_r2": round(best_score, 6),
        "selection_history": history,
    }


# ── 4. Backward elimination ───────────────────────────────────────────────────

def backward_elimination(
    df: pd.DataFrame,
    target: str,
    max_features: int = 20,
    cv: int = 3,
    exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Backward elimination: start with all features, iteratively remove the one
    whose removal least hurts (or most improves) cross-validated R².
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    candidates = _numeric_features(df, exclude=[target] + (exclude or []))
    X_all = df[candidates].fillna(df[candidates].mean())
    y = df[target].fillna(df[target].mean())

    remaining = candidates.copy()
    history: list[dict] = []

    while len(remaining) > max_features:
        step_scores: dict[str, float] = {}
        for col in remaining:
            cols = [c for c in remaining if c != col]
            if not cols:
                break
            folds = min(cv, len(X_all))
            score = cross_val_score(
                LinearRegression(), X_all[cols], y, cv=folds, scoring="r2"
            ).mean()
            step_scores[col] = round(float(score), 6)

        # Remove the feature whose absence yields the best score
        removed = max(step_scores, key=step_scores.get)
        remaining.remove(removed)
        history.append({"removed": removed, "r2_without": step_scores[removed]})

    return {
        "method": "backward_elimination",
        "selected": remaining,
        "elimination_history": history,
    }


# ── 5. Recursive feature elimination (RFE) ───────────────────────────────────

def recursive_elimination(
    model: Any,
    df: pd.DataFrame,
    target: str,
    n_features: int = 10,
    exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    sklearn RFE wrapper. `model` must expose `coef_` or `feature_importances_`
    (e.g. LinearRegression, RandomForestClassifier).
    """
    from sklearn.feature_selection import RFE

    candidates = _numeric_features(df, exclude=[target] + (exclude or []))
    X = df[candidates].fillna(df[candidates].mean()).values
    y = df[target].fillna(df[target].mean()).values

    n_select = min(n_features, len(candidates))
    rfe = RFE(estimator=model, n_features_to_select=n_select)
    rfe.fit(X, y)

    selected = [candidates[i] for i, s in enumerate(rfe.support_) if s]
    rankings = {candidates[i]: int(rfe.ranking_[i]) for i in range(len(candidates))}

    return {
        "method": "recursive_elimination",
        "selected": selected,
        "rankings": dict(sorted(rankings.items(), key=lambda x: x[1])),
    }


# ── 6. Information value (IV) ─────────────────────────────────────────────────

def information_value(
    df: pd.DataFrame,
    target_col: str,
    bins: int = 10,
    exclude: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) per feature.
    Target must be binary (0/1). Higher IV = stronger predictor.

    IV thresholds:
      < 0.02  → useless
      0.02–0.1 → weak
      0.1–0.3  → medium
      0.3–0.5  → strong
      > 0.5   → suspicious (possible leakage)
    """
    target = df[target_col]
    total_events = target.sum()
    total_non_events = len(target) - total_events

    if total_events == 0 or total_non_events == 0:
        raise ValueError("target_col must be binary with both classes present.")

    candidates = _numeric_features(df, exclude=[target_col] + (exclude or []))
    results: dict[str, dict] = {}

    for col in candidates:
        try:
            binned = pd.qcut(df[col], q=bins, duplicates="drop")
        except Exception:
            continue

        iv = 0.0
        woe_bins: dict[str, float] = {}

        for bin_label in binned.cat.categories:
            mask = binned == bin_label
            events = float(target[mask].sum())
            non_events = float(mask.sum() - events)

            dist_e = events / total_events
            dist_ne = non_events / total_non_events

            if dist_e == 0 or dist_ne == 0:
                continue

            woe = np.log(dist_e / dist_ne)
            iv += (dist_e - dist_ne) * woe
            woe_bins[str(bin_label)] = round(woe, 4)

        strength = (
            "useless" if iv < 0.02
            else "weak" if iv < 0.1
            else "medium" if iv < 0.3
            else "strong" if iv < 0.5
            else "suspicious"
        )
        results[col] = {"iv": round(iv, 4), "strength": strength, "woe": woe_bins}

    return dict(sorted(results.items(), key=lambda x: x[1]["iv"], reverse=True))
