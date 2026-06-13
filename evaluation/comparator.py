"""
Model comparison — ranks training runs, tests statistical significance,
and identifies the best model given a user-specified priority metric.

Works entirely from stored metrics dicts so no models need to be reloaded.
"""

from typing import Any, Literal, Optional

import numpy as np

Metric = Literal["accuracy", "f1", "roc_auc", "precision", "recall", "r2", "mae", "rmse"]

# Lower is better for these metrics
_LOWER_IS_BETTER = {"mae", "rmse", "mape"}


# ── ranking ───────────────────────────────────────────────────────────────────

def rank_runs(
    runs: list[dict],
    metric: str = "accuracy",
) -> list[dict]:
    """
    Sort runs by `metric`. Returns a list of run dicts with `rank` and `metric_value`
    added. Runs missing the metric are placed at the end.
    """
    lower_better = metric in _LOWER_IS_BETTER

    def sort_key(r: dict) -> float:
        val = (r.get("metrics") or {}).get(metric)
        if val is None:
            return float("inf") if lower_better else float("-inf")
        return float(val)

    ranked = sorted(runs, key=sort_key, reverse=not lower_better)
    for i, r in enumerate(ranked):
        r = dict(r)
        ranked[i] = {
            **r,
            "rank": i + 1,
            "metric_used": metric,
            "metric_value": (r.get("metrics") or {}).get(metric),
        }
    return ranked


# ── pairwise statistical significance ────────────────────────────────────────

def mcnemar_test(
    cm_a: list[list[int]],
    cm_b: list[list[int]],
    n_test: int,
    alpha: float = 0.05,
) -> dict:
    """
    McNemar's test for comparing two classifiers on the same test set.
    Uses the confusion matrix totals to reconstruct discordant pair counts.

    Discordant pairs:
      n01 = A wrong, B right  ≈ B_correct − A_correct
      n10 = A right, B wrong  ≈ A_correct − B_correct
    (exact counts require per-sample predictions — this is an approximation)
    """
    from scipy.stats import chi2

    correct_a = sum(cm_a[i][i] for i in range(len(cm_a)))
    correct_b = sum(cm_b[i][i] for i in range(len(cm_b)))
    n01 = max(0, correct_b - correct_a)   # B right, A wrong
    n10 = max(0, correct_a - correct_b)   # A right, B wrong

    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "note": "No discordant pairs — models make identical errors."}

    # McNemar statistic with continuity correction (Edwards)
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "discordant_n01": int(n01),
        "discordant_n10": int(n10),
    }


def t_test_metrics(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """
    Welch's t-test comparing two lists of metric values (e.g. from CV folds).
    Use when you have multiple evaluation scores per model.
    """
    from scipy.stats import ttest_ind

    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    stat, p_value = ttest_ind(a, b, equal_var=False)

    return {
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "diff":   round(float(a.mean() - b.mean()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
    }


# ── practical significance ─────────────────────────────────────────────────────

def is_practically_significant(
    val_a: float,
    val_b: float,
    metric: str,
    threshold: float = 0.01,
) -> bool:
    """
    True if the difference between two metric values exceeds `threshold`.
    Default threshold of 0.01 (1%) is a common rule of thumb for accuracy/F1.
    """
    return abs(val_a - val_b) > threshold


# ── full comparison ───────────────────────────────────────────────────────────

def compare_runs(
    runs: list[dict],
    primary_metric: str = "accuracy",
    practical_threshold: float = 0.01,
    alpha: float = 0.05,
) -> dict:
    """
    Compare all runs: rank them, run pairwise significance tests between the
    top two, and return a structured leaderboard with a winner recommendation.

    `runs`: list of dicts with keys: run_id, model_type, metrics
    """
    if not runs:
        return {"error": "No runs to compare."}

    ranked = rank_runs(runs, metric=primary_metric)

    pairwise_tests = []
    if len(ranked) >= 2:
        top, second = ranked[0], ranked[1]

        # Statistical test (McNemar if confusion matrices available)
        cm_top = (top.get("metrics") or {}).get("confusion_matrix")
        cm_sec = (second.get("metrics") or {}).get("confusion_matrix")
        n_test = _infer_n_test(top.get("metrics") or {})

        if cm_top and cm_sec and n_test:
            sig_test = mcnemar_test(cm_top, cm_sec, n_test, alpha)
            sig_test["model_a"] = top["model_type"]
            sig_test["model_b"] = second["model_type"]
            pairwise_tests.append(sig_test)

        val_top = top.get("metric_value") or 0.0
        val_sec = second.get("metric_value") or 0.0
        practical = is_practically_significant(val_top, val_sec, primary_metric, practical_threshold)
    else:
        practical = True

    winner = _choose_winner(ranked, primary_metric, practical_threshold)

    return {
        "primary_metric": primary_metric,
        "leaderboard": ranked,
        "pairwise_significance": pairwise_tests,
        "winner": winner,
        "total_runs": len(runs),
    }


# ── best model selector ───────────────────────────────────────────────────────

def find_best_model(
    runs: list[dict],
    primary_metric: str = "accuracy",
    tiebreak_metric: str = "training_time_s",
    practical_threshold: float = 0.01,
) -> dict:
    """
    Return the best run. If the top two models are within `practical_threshold`,
    prefer the one with better `tiebreak_metric` (usually faster or simpler).

    Adds a human-readable recommendation.
    """
    if not runs:
        return {"error": "No runs available."}

    ranked = rank_runs(runs, metric=primary_metric)
    best = ranked[0]

    if len(ranked) >= 2:
        second = ranked[1]
        val_best = best.get("metric_value") or 0.0
        val_sec = second.get("metric_value") or 0.0

        if not is_practically_significant(val_best, val_sec, primary_metric, practical_threshold):
            # Tie on primary — use tiebreak
            tb_best = (best.get("metrics") or {}).get(tiebreak_metric) or float("inf")
            tb_sec = (second.get("metrics") or {}).get(tiebreak_metric) or float("inf")

            if tiebreak_metric not in _LOWER_IS_BETTER:
                best = best if tb_best >= tb_sec else second
            else:
                best = best if tb_best <= tb_sec else second

    model_type = best["model_type"]
    metric_val = best.get("metric_value")
    t_time = (best.get("metrics") or {}).get("training_time_s")

    recommendation = (
        f"Use {model_type}: {primary_metric}={metric_val}"
        + (f", trained in {t_time:.2f}s" if t_time else "")
        + ("." if not t_time else ".")
    )

    return {
        "best_run": best,
        "primary_metric": primary_metric,
        "recommendation": recommendation,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _infer_n_test(metrics: dict) -> Optional[int]:
    cm = metrics.get("confusion_matrix")
    if cm:
        return int(sum(sum(row) for row in cm))
    return None


def _choose_winner(ranked: list[dict], metric: str, threshold: float) -> dict:
    if not ranked:
        return {}
    best = ranked[0]
    if len(ranked) == 1:
        return best
    second = ranked[1]
    val_b = best.get("metric_value") or 0.0
    val_s = second.get("metric_value") or 0.0
    margin = round(abs(val_b - val_s), 4)
    return {
        **best,
        "margin_over_second": margin,
        "clearly_better": is_practically_significant(val_b, val_s, metric, threshold),
    }
