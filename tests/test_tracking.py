"""
MLflow integration tests.
Each test uses an isolated SQLite tracking URI so tests never affect each other
or a running MLflow server.
"""

import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import mlflow

from models.trainer import TrainResult, train_model, train_multiple
from tracking.mlflow_integration import MLflowTracker


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tracker(tmp_path, monkeypatch):
    """Isolated MLflowTracker backed by a temporary SQLite DB.
    Sets MLFLOW_TRACKING_URI so that train_model() picks up the same URI
    when it creates its own internal MLflowTracker instance.
    """
    uri = f"sqlite:///{tmp_path}/mlflow_test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    t = MLflowTracker(tracking_uri=uri)
    yield t
    mlflow.end_run()


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    return X[:120], X[120:], y[:120], y[120:]


@pytest.fixture
def sample_result(clf_data):
    X_tr, X_val, y_tr, y_val = clf_data
    return train_model(
        "random_forest", X_tr, y_tr, X_val, y_val,
        hyperparams={"n_estimators": 5, "max_depth": 3},
    )


# ─── experiment management ────────────────────────────────────────────────────

class TestExperimentManagement:
    def test_creates_new_experiment(self, tracker):
        exp_id = tracker.get_or_create_experiment(project_id=1, project_name="test_proj")
        assert exp_id is not None
        assert len(exp_id) > 0

    def test_idempotent_experiment_creation(self, tracker):
        id1 = tracker.get_or_create_experiment(1, "same_project")
        id2 = tracker.get_or_create_experiment(1, "same_project")
        assert id1 == id2

    def test_different_projects_get_different_experiments(self, tracker):
        id1 = tracker.get_or_create_experiment(1, "project_alpha")
        id2 = tracker.get_or_create_experiment(2, "project_beta")
        assert id1 != id2

    def test_get_experiment_returns_metadata(self, tracker):
        exp_id = tracker.get_or_create_experiment(1, "meta_test")
        exp = tracker.get_experiment(exp_id)
        assert exp is not None
        assert exp["experiment_id"] == exp_id
        assert "project_1" in exp["name"]

    def test_get_nonexistent_experiment_returns_none(self, tracker):
        result = tracker.get_experiment("99999999")
        assert result is None


# ─── run logging ─────────────────────────────────────────────────────────────

class TestExperimentLogging:
    def test_log_run_returns_run_id(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(1, "logging_test")
        run_id = tracker.log_run(
            result=sample_result,
            experiment_id=exp_id,
            feature_set_name="v1",
            project_id=1,
            db_run_id=42,
        )
        assert run_id is not None
        assert len(run_id) > 0

    def test_logged_params_match_hyperparams(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(1, "params_test")
        run_id = tracker.log_run(sample_result, exp_id, "v1", 1)
        run = tracker.get_run(run_id)
        assert run["params"]["n_estimators"] == "5"
        assert run["params"]["max_depth"] == "3"

    def test_logged_metrics_include_accuracy(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(1, "metrics_test")
        run_id = tracker.log_run(sample_result, exp_id, "v1", 1)
        run = tracker.get_run(run_id)
        assert "accuracy" in run["metrics"]
        assert 0.0 <= run["metrics"]["accuracy"] <= 1.0

    def test_logged_metrics_include_training_time(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(1, "timing_test")
        run_id = tracker.log_run(sample_result, exp_id, "v1", 1)
        run = tracker.get_run(run_id)
        assert "training_time_s" in run["metrics"]
        assert run["metrics"]["training_time_s"] > 0

    def test_tags_contain_model_type(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(1, "tags_test")
        run_id = tracker.log_run(sample_result, exp_id, "features_v2", 1, db_run_id=7)
        run = tracker.get_run(run_id)
        assert run["tags"]["model_type"] == "random_forest"
        assert run["tags"]["feature_set"] == "features_v2"
        assert run["tags"]["db_run_id"] == "7"

    def test_confusion_matrix_not_logged_as_metric(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(1, "skip_keys_test")
        run_id = tracker.log_run(sample_result, exp_id, "v1", 1)
        run = tracker.get_run(run_id)
        assert "confusion_matrix" not in run["metrics"]
        assert "per_class" not in run["metrics"]

    def test_train_model_with_mlflow_sets_run_id(self, tracker, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        exp_id = tracker.get_or_create_experiment(1, "trainer_test")
        result = train_model(
            "logistic_regression", X_tr, y_tr, X_val, y_val,
            hyperparams={"C": 0.5},
            mlflow_experiment_id=exp_id,
            mlflow_feature_set_name="v1",
            mlflow_project_id=1,
        )
        assert result.mlflow_run_id is not None

    def test_mlflow_run_id_in_metrics_dict(self, tracker, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        exp_id = tracker.get_or_create_experiment(1, "metrics_dict_test")
        result = train_model(
            "logistic_regression", X_tr, y_tr, X_val, y_val,
            mlflow_experiment_id=exp_id,
            mlflow_feature_set_name="v1",
            mlflow_project_id=1,
        )
        d = result.to_metrics_dict()
        assert "mlflow_run_id" in d
        assert d["mlflow_run_id"] == result.mlflow_run_id


# ─── run comparison ───────────────────────────────────────────────────────────

class TestRunComparison:
    def test_multiple_runs_all_logged(self, tracker, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        exp_id = tracker.get_or_create_experiment(2, "comparison_test")

        configs = [
            {"type": "logistic_regression", "hyperparams": {"C": 1.0}},
            {"type": "random_forest",       "hyperparams": {"n_estimators": 5}},
        ]
        train_multiple(
            configs, X_tr, y_tr, X_val, y_val,
            mlflow_experiment_id=exp_id,
            mlflow_feature_set_name="v1",
            mlflow_project_id=2,
        )

        runs = tracker.get_runs(exp_id)
        assert len(runs) == 2

    def test_get_runs_ordered_by_accuracy(self, tracker, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        exp_id = tracker.get_or_create_experiment(3, "order_test")

        configs = [
            {"type": "logistic_regression", "hyperparams": {}},
            {"type": "random_forest",       "hyperparams": {"n_estimators": 5}},
        ]
        train_multiple(
            configs, X_tr, y_tr, X_val, y_val,
            mlflow_experiment_id=exp_id,
            mlflow_feature_set_name="v1",
            mlflow_project_id=3,
        )

        runs = tracker.get_runs(exp_id, order_by="metrics.accuracy DESC")
        accs = [r["metrics"].get("accuracy", 0) for r in runs if "accuracy" in r["metrics"]]
        assert accs == sorted(accs, reverse=True)

    def test_get_best_run_has_highest_accuracy(self, tracker, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        exp_id = tracker.get_or_create_experiment(4, "best_test")

        configs = [
            {"type": "logistic_regression", "hyperparams": {}},
            {"type": "random_forest",       "hyperparams": {"n_estimators": 10}},
        ]
        train_multiple(
            configs, X_tr, y_tr, X_val, y_val,
            mlflow_experiment_id=exp_id,
            mlflow_feature_set_name="v1",
            mlflow_project_id=4,
        )

        runs = tracker.get_runs(exp_id, order_by="metrics.accuracy DESC")
        best = tracker.get_best_run(exp_id, metric="accuracy")

        top_acc = runs[0]["metrics"].get("accuracy", 0)
        assert best is not None
        assert best["metrics"].get("accuracy") == pytest.approx(top_acc, abs=1e-4)

    def test_ui_url_contains_experiment_id(self, tracker):
        exp_id = tracker.get_or_create_experiment(5, "url_test")
        url = tracker.ui_url(experiment_id=exp_id)
        assert exp_id in url

    def test_ui_url_contains_run_id(self, tracker, sample_result):
        exp_id = tracker.get_or_create_experiment(6, "run_url_test")
        run_id = tracker.log_run(sample_result, exp_id, "v1", 6)
        url = tracker.ui_url(experiment_id=exp_id, run_id=run_id)
        assert run_id in url

    def test_train_multiple_without_mlflow_still_returns_results(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        configs = [{"type": "logistic_regression", "hyperparams": {}}]
        results = train_multiple(configs, X_tr, y_tr, X_val, y_val)
        assert len(results) == 1
        assert results[0].mlflow_run_id is None
