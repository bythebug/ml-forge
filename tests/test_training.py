import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from models.model_definitions import DEFAULTS, build_model, list_model_types
from models.trainer import TrainResult, load_model, save_model, train_model, train_multiple


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    mid = 160
    return X[:mid], X[mid:], y[:mid], y[mid:]


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
    mid = 160
    return X[:mid], X[mid:], y[:mid], y[mid:]


# ─── model building ───────────────────────────────────────────────────────────

class TestModelBuilding:
    def test_all_model_types_build(self):
        for model_type in list_model_types():
            if model_type == "xgboost":
                pytest.importorskip("xgboost")
            model = build_model(model_type)
            assert model is not None

    def test_hyperparams_override_defaults(self):
        model = build_model("random_forest", hyperparams={"n_estimators": 7})
        assert model.n_estimators == 7

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model("transformer")


# ─── logistic regression ──────────────────────────────────────────────────────

class TestLogisticRegression:
    def test_trains_and_returns_result(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("logistic_regression", X_tr, y_tr, X_val, y_val)
        assert isinstance(result, TrainResult)
        assert result.model_type == "logistic_regression"

    def test_accuracy_above_chance(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("logistic_regression", X_tr, y_tr, X_val, y_val)
        assert result.metrics["accuracy"] > 0.5

    def test_metrics_contain_expected_keys(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("logistic_regression", X_tr, y_tr, X_val, y_val)
        for key in ("accuracy", "f1", "precision", "recall"):
            assert key in result.metrics

    def test_hyperparams_stored_in_result(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("logistic_regression", X_tr, y_tr, X_val, y_val,
                             hyperparams={"C": 0.5})
        assert result.hyperparams["C"] == 0.5

    def test_l1_penalty(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model(
            "logistic_regression", X_tr, y_tr, X_val, y_val,
            hyperparams={"penalty": "l1", "solver": "liblinear"}
        )
        assert result.metrics["accuracy"] > 0.5


# ─── SVM ─────────────────────────────────────────────────────────────────────

class TestSVM:
    def test_rbf_kernel_trains(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("svm", X_tr, y_tr, X_val, y_val)
        assert result.metrics["accuracy"] > 0.5

    def test_linear_kernel(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("svm", X_tr, y_tr, X_val, y_val,
                             hyperparams={"kernel": "linear"})
        assert "accuracy" in result.metrics

    def test_roc_auc_computed_for_binary(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("svm", X_tr, y_tr, X_val, y_val)
        assert "roc_auc" in result.metrics
        assert 0.0 <= result.metrics["roc_auc"] <= 1.0

    def test_training_time_recorded(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("svm", X_tr, y_tr, X_val, y_val)
        assert result.training_time_s > 0


# ─── random forest ────────────────────────────────────────────────────────────

class TestRandomForest:
    def test_trains_with_defaults(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("random_forest", X_tr, y_tr, X_val, y_val)
        assert result.metrics["accuracy"] > 0.5

    def test_custom_n_estimators(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("random_forest", X_tr, y_tr, X_val, y_val,
                             hyperparams={"n_estimators": 10})
        assert result.hyperparams["n_estimators"] == 10

    def test_max_depth_limits_tree(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("random_forest", X_tr, y_tr, X_val, y_val,
                             hyperparams={"max_depth": 2, "n_estimators": 10})
        for estimator in result.model.estimators_:
            assert estimator.get_depth() <= 2

    def test_regression_task(self, reg_data):
        X_tr, X_val, y_tr, y_val = reg_data
        result = train_model("random_forest", X_tr, y_tr, X_val, y_val, task="regression")
        assert "r2" in result.metrics
        assert "mae" in result.metrics


# ─── XGBoost ─────────────────────────────────────────────────────────────────

class TestXGBoost:
    def test_trains_classification(self, clf_data):
        pytest.importorskip("xgboost")
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("xgboost", X_tr, y_tr, X_val, y_val,
                             hyperparams={"n_estimators": 20})
        assert result.metrics["accuracy"] > 0.5

    def test_learning_rate_stored(self, clf_data):
        pytest.importorskip("xgboost")
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("xgboost", X_tr, y_tr, X_val, y_val,
                             hyperparams={"learning_rate": 0.05, "n_estimators": 20})
        assert result.hyperparams["learning_rate"] == 0.05

    def test_not_installed_raises_import_error(self, clf_data, monkeypatch):
        import builtins
        real_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("mocked missing xgboost")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="xgboost"):
            build_model("xgboost")


# ─── neural network ───────────────────────────────────────────────────────────

class TestNeuralNetwork:
    def test_trains_with_defaults(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("neural_network", X_tr, y_tr, X_val, y_val,
                             hyperparams={"hidden_layer_sizes": (32,), "max_iter": 50})
        assert result.metrics["accuracy"] > 0.4

    def test_custom_architecture(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model(
            "neural_network", X_tr, y_tr, X_val, y_val,
            hyperparams={"hidden_layer_sizes": (64, 32), "max_iter": 50}
        )
        assert result.model.hidden_layer_sizes == (64, 32)

    def test_relu_and_tanh_activations(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        for activation in ("relu", "tanh"):
            result = train_model(
                "neural_network", X_tr, y_tr, X_val, y_val,
                hyperparams={"activation": activation, "max_iter": 30,
                             "hidden_layer_sizes": (16,), "early_stopping": False}
            )
            assert "accuracy" in result.metrics


# ─── model serialization ──────────────────────────────────────────────────────

class TestModelSerialization:
    def test_save_and_load_preserves_predictions(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("random_forest", X_tr, y_tr, X_val, y_val,
                             hyperparams={"n_estimators": 10})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            save_model(result.model, path)
            loaded = load_model(path)
            np.testing.assert_array_equal(
                result.model.predict(X_val),
                loaded.predict(X_val),
            )

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")

    def test_save_creates_parent_directory(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("logistic_regression", X_tr, y_tr, X_val, y_val)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "model.joblib"
            save_model(result.model, path)
            assert path.exists()


# ─── multi-model training ─────────────────────────────────────────────────────

class TestTrainMultiple:
    def test_trains_all_configs(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        configs = [
            {"type": "logistic_regression", "hyperparams": {}},
            {"type": "random_forest", "hyperparams": {"n_estimators": 10}},
        ]
        results = train_multiple(configs, X_tr, y_tr, X_val, y_val)
        assert len(results) == 2

    def test_results_sorted_by_accuracy(self, clf_data):
        X_tr, X_val, y_tr, y_val = clf_data
        configs = [
            {"type": "logistic_regression", "hyperparams": {}},
            {"type": "random_forest", "hyperparams": {"n_estimators": 10}},
        ]
        results = train_multiple(configs, X_tr, y_tr, X_val, y_val)
        accuracies = [r.metrics["accuracy"] for r in results]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_to_metrics_dict_serializable(self, clf_data):
        import json
        X_tr, X_val, y_tr, y_val = clf_data
        result = train_model("logistic_regression", X_tr, y_tr, X_val, y_val)
        d = result.to_metrics_dict()
        assert d["status"] == "completed"
        assert "training_time_s" in d
        json.dumps(d)  # must be JSON-serializable for JSONB storage
