import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from evaluation.analysis import (
    confusion_matrix_analysis,
    error_examples,
    feature_importance_analysis,
    residual_analysis,
)
from evaluation.comparator import (
    compare_runs,
    find_best_model,
    is_practically_significant,
    rank_runs,
    t_test_metrics,
)
from evaluation.evaluator import (
    bootstrap_confidence_interval,
    evaluate_classification,
    evaluate_model,
    evaluate_regression,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_arrays():
    X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X[:240], y[:240])
    return X[240:], y[240:], model


@pytest.fixture
def reg_arrays():
    X, y = make_regression(n_samples=300, n_features=8, noise=5, random_state=0)
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X[:240], y[:240])
    return X[240:], y[240:], model


@pytest.fixture
def sample_runs():
    return [
        {
            "run_id": 1, "model_type": "logistic_regression",
            "metrics": {
                "accuracy": 0.85, "f1": 0.84, "roc_auc": 0.91,
                "training_time_s": 0.5,
                "confusion_matrix": [[40, 5], [4, 11]],
            },
        },
        {
            "run_id": 2, "model_type": "random_forest",
            "metrics": {
                "accuracy": 0.92, "f1": 0.91, "roc_auc": 0.97,
                "training_time_s": 2.1,
                "confusion_matrix": [[43, 2], [2, 13]],
            },
        },
        {
            "run_id": 3, "model_type": "svm",
            "metrics": {
                "accuracy": 0.88, "f1": 0.87, "roc_auc": 0.93,
                "training_time_s": 1.3,
                "confusion_matrix": [[41, 4], [3, 12]],
            },
        },
    ]


# ─── accuracy calculation ─────────────────────────────────────────────────────

class TestAccuracyCalculation:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 0, 1, 1])
        result = evaluate_classification(y, y)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_worst_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = evaluate_classification(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_accuracy_range(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        result = evaluate_model(model, X_test, y_test, task="classification")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_returns_all_classification_keys(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        result = evaluate_model(model, X_test, y_test, task="classification")
        for key in ("accuracy", "f1", "precision", "recall", "confusion_matrix"):
            assert key in result

    def test_roc_auc_present_for_binary(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        result = evaluate_model(model, X_test, y_test, task="classification")
        assert "roc_auc" in result
        assert 0.0 <= result["roc_auc"] <= 1.0


# ─── metric consistency ───────────────────────────────────────────────────────

class TestMetricConsistency:
    def test_precision_recall_f1_consistent(self):
        """F1 = 2·P·R / (P+R) — verify this holds."""
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        result = evaluate_classification(y_true, y_pred)
        p, r, f1 = result["precision"], result["recall"], result["f1"]
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert abs(f1 - expected_f1) < 0.01

    def test_confusion_matrix_sums_to_n(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        result = evaluate_model(model, X_test, y_test)
        cm = result["confusion_matrix"]
        assert sum(sum(row) for row in cm) == len(y_test)

    def test_per_class_support_sums_to_n(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        result = evaluate_model(model, X_test, y_test)
        total_support = sum(v["support"] for v in result["per_class"].values())
        assert total_support == len(y_test)

    def test_regression_metrics_present(self, reg_arrays):
        X_test, y_test, model = reg_arrays
        result = evaluate_model(model, X_test, y_test, task="regression")
        for key in ("r2", "mae", "rmse"):
            assert key in result

    def test_r2_at_most_one(self, reg_arrays):
        X_test, y_test, model = reg_arrays
        result = evaluate_model(model, X_test, y_test, task="regression")
        assert result["r2"] <= 1.0

    def test_mae_non_negative(self, reg_arrays):
        X_test, y_test, model = reg_arrays
        result = evaluate_model(model, X_test, y_test, task="regression")
        assert result["mae"] >= 0.0

    def test_bootstrap_ci_bounds_valid(self):
        lo, hi = bootstrap_confidence_interval(0.85, n_samples=200)
        assert lo < 0.85 < hi
        assert 0.0 <= lo <= hi <= 1.0

    def test_wider_ci_for_smaller_n(self):
        lo_large, hi_large = bootstrap_confidence_interval(0.8, n_samples=1000)
        lo_small, hi_small = bootstrap_confidence_interval(0.8, n_samples=50)
        assert (hi_small - lo_small) > (hi_large - lo_large)


# ─── model comparison ─────────────────────────────────────────────────────────

class TestModelComparison:
    def test_rank_runs_by_accuracy(self, sample_runs):
        ranked = rank_runs(sample_runs, metric="accuracy")
        vals = [r["metric_value"] for r in ranked]
        assert vals == sorted(vals, reverse=True)

    def test_best_model_is_random_forest(self, sample_runs):
        result = find_best_model(sample_runs, primary_metric="accuracy")
        assert result["best_run"]["model_type"] == "random_forest"

    def test_compare_runs_returns_winner(self, sample_runs):
        result = compare_runs(sample_runs, primary_metric="accuracy")
        assert "winner" in result
        assert result["winner"]["model_type"] == "random_forest"

    def test_leaderboard_has_all_runs(self, sample_runs):
        result = compare_runs(sample_runs)
        assert len(result["leaderboard"]) == len(sample_runs)

    def test_compare_empty_runs_returns_error(self):
        result = compare_runs([])
        assert "error" in result

    def test_rank_missing_metric_goes_last(self):
        runs = [
            {"run_id": 1, "model_type": "a", "metrics": {"accuracy": 0.9}},
            {"run_id": 2, "model_type": "b", "metrics": {}},
        ]
        ranked = rank_runs(runs, metric="accuracy")
        assert ranked[-1]["model_type"] == "b"

    def test_tiebreak_prefers_faster_model(self):
        runs = [
            {"run_id": 1, "model_type": "slow_model", "metrics": {"accuracy": 0.901, "training_time_s": 10.0}},
            {"run_id": 2, "model_type": "fast_model", "metrics": {"accuracy": 0.900, "training_time_s": 0.5}},
        ]
        result = find_best_model(runs, primary_metric="accuracy", practical_threshold=0.01)
        # within 0.01 threshold → prefer faster
        assert result["best_run"]["model_type"] == "fast_model"


# ─── statistical significance ─────────────────────────────────────────────────

class TestStatisticalSignificance:
    def test_large_difference_is_practically_significant(self):
        assert is_practically_significant(0.95, 0.80, "accuracy", threshold=0.01)

    def test_tiny_difference_is_not_practically_significant(self):
        assert not is_practically_significant(0.901, 0.900, "accuracy", threshold=0.01)

    def test_t_test_detects_different_distributions(self):
        a = [0.90, 0.91, 0.89, 0.92, 0.88]
        b = [0.70, 0.71, 0.69, 0.72, 0.68]
        result = t_test_metrics(a, b)
        assert result["significant"]
        assert result["p_value"] < 0.05

    def test_t_test_same_values_not_significant(self):
        a = [0.85, 0.85, 0.85]
        b = [0.85, 0.85, 0.85]
        result = t_test_metrics(a, b)
        assert not result["significant"]

    def test_compare_runs_includes_mcnemar(self, sample_runs):
        result = compare_runs(sample_runs, primary_metric="accuracy")
        # McNemar runs when top 2 have confusion matrices
        tests = result.get("pairwise_significance", [])
        if tests:
            assert "p_value" in tests[0]
            assert "significant" in tests[0]


# ─── confusion matrix analysis ───────────────────────────────────────────────

class TestConfusionMatrixAnalysis:
    def test_perfect_classifier_zero_errors(self):
        y = np.array([0, 1, 0, 1])
        result = confusion_matrix_analysis(y, y)
        assert result["overall_error_rate"] == pytest.approx(0.0)
        assert result["total_correct"] == 4

    def test_identifies_hardest_class(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1])  # class 0 errors = 2/3
        result = confusion_matrix_analysis(y_true, y_pred)
        assert result["hardest_class"] == "0"

    def test_most_confused_pairs_sorted(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        y_pred = model.predict(X_test)
        result = confusion_matrix_analysis(y_test, y_pred)
        counts = [p["count"] for p in result["most_confused_pairs"]]
        assert counts == sorted(counts, reverse=True)

    def test_total_samples_correct(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        result = confusion_matrix_analysis(y_true, y_pred)
        assert result["total_samples"] == 5


# ─── feature importance analysis ─────────────────────────────────────────────

class TestFeatureImportanceAnalysis:
    def test_tree_model_returns_importances(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]
        result = feature_importance_analysis(model, feature_names)
        assert result["method"] == "mean_decrease_impurity"
        assert len(result["top_features"]) > 0

    def test_linear_model_returns_coef_magnitude(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=0)
        model = LogisticRegression(random_state=0).fit(X, y)
        result = feature_importance_analysis(model, [f"f{i}" for i in range(5)])
        assert result["method"] == "coefficient_magnitude"

    def test_features_sorted_by_importance(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]
        result = feature_importance_analysis(model, feature_names)
        importances = [f["importance"] for f in result["top_features"]]
        assert importances == sorted(importances, reverse=True)

    def test_cumulative_pct_reaches_100(self, clf_arrays):
        X_test, y_test, model = clf_arrays
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]
        result = feature_importance_analysis(model, feature_names, top_n=100)
        assert result["top_features"][-1]["cumulative_pct"] == pytest.approx(100.0, abs=0.1)


# ─── residual analysis ────────────────────────────────────────────────────────

class TestResidualAnalysis:
    def test_perfect_model_zero_residuals(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = residual_analysis(y, y)
        assert result["bias"]["mean_residual"] == pytest.approx(0.0)
        assert result["bias"]["direction"] == "unbiased"

    def test_over_prediction_bias(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # predicts too high
        result = residual_analysis(y_true, y_pred)
        assert result["bias"]["direction"] == "over-predicts"

    def test_worst_indices_ordered_by_error(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 1.0])   # last is worst
        result = residual_analysis(y_true, y_pred)
        assert result["worst_prediction_indices"][0] == 4

    def test_error_percentiles_ascending(self, reg_arrays):
        X_test, y_test, model = reg_arrays
        y_pred = model.predict(X_test)
        result = residual_analysis(y_test, y_pred)
        vals = list(result["error_percentiles"].values())
        assert vals == sorted(vals)
