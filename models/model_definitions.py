"""
Model factory for 5 model families.
Each model type has typed default hyperparameters and a build function.
Supports classification (default) and regression via `task` parameter.
"""

from typing import Any, Literal

import numpy as np

ModelType = Literal[
    "logistic_regression", "svm", "random_forest", "xgboost", "neural_network"
]
Task = Literal["classification", "regression"]

# ── default hyperparameters ───────────────────────────────────────────────────

DEFAULTS: dict[ModelType, dict[str, Any]] = {
    "logistic_regression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,   # needed for predict_proba / ROC-AUC
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
    },
    "neural_network": {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "learning_rate": "adaptive",
        "learning_rate_init": 0.001,
        "alpha": 0.0001,         # L2 regularization
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
    },
}

HYPERPARAMETER_DOCS: dict[ModelType, dict[str, str]] = {
    "logistic_regression": {
        "penalty":    "Regularisation norm: 'l1' (sparse), 'l2' (default), 'elasticnet', 'none'",
        "C":          "Inverse regularisation strength — smaller = stronger regularisation",
        "solver":     "'lbfgs' (default, L2), 'liblinear' (L1/L2 small data), 'saga' (L1/elasticnet large data)",
        "max_iter":   "Max iterations for solver convergence",
    },
    "svm": {
        "kernel":     "'rbf' (non-linear default), 'linear', 'poly', 'sigmoid'",
        "C":          "Margin softness — large C = hard margin (low bias, high variance)",
        "gamma":      "'scale' (1/n_features*var), 'auto' (1/n_features), or float — RBF bandwidth",
    },
    "random_forest": {
        "n_estimators":       "Number of trees — more = lower variance, diminishing returns after ~300",
        "max_depth":          "Max tree depth — None = fully grown (may overfit)",
        "min_samples_split":  "Minimum samples to split a node — higher = more regularisation",
        "min_samples_leaf":   "Minimum samples in a leaf — smooths predictions",
        "max_features":       "'sqrt' (classification default), 'log2', float — feature subset per split",
    },
    "xgboost": {
        "n_estimators":     "Number of boosting rounds",
        "max_depth":        "Max tree depth per round — 3-6 typical",
        "learning_rate":    "Step shrinkage (eta) — lower = slower but more robust",
        "subsample":        "Row sampling per round — prevents overfitting",
        "colsample_bytree": "Column sampling per round — prevents overfitting",
    },
    "neural_network": {
        "hidden_layer_sizes": "Tuple of neurons per hidden layer e.g. (128, 64)",
        "activation":         "'relu' (default), 'tanh', 'logistic'",
        "learning_rate":      "'adaptive' reduces LR when loss plateaus",
        "alpha":              "L2 penalty coefficient — higher = more regularisation",
        "early_stopping":     "Stop when validation score stops improving",
    },
}


# ── factory ───────────────────────────────────────────────────────────────────

def build_model(
    model_type: ModelType,
    hyperparams: dict[str, Any] | None = None,
    task: Task = "classification",
):
    """
    Build and return an unfitted sklearn-compatible estimator.
    Merges user `hyperparams` over the model's defaults.
    """
    if model_type not in DEFAULTS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(DEFAULTS.keys())}"
        )
    params = {**DEFAULTS[model_type], **(hyperparams or {})}

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression, LinearRegression
        return LogisticRegression(**params) if task == "classification" else LinearRegression()

    if model_type == "svm":
        from sklearn.svm import SVC, SVR
        # SVR doesn't support probability or random_state
        if task == "regression":
            svm_params = {k: v for k, v in params.items()
                         if k in ("kernel", "C", "gamma")}
            return SVR(**svm_params)
        return SVC(**{k: v for k, v in params.items() if k != "random_state" or True})

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(**params)

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from exc
        cls = XGBClassifier if task == "classification" else XGBRegressor
        xgb_params = {k: v for k, v in params.items() if k != "random_state"}
        xgb_params["seed"] = params.get("random_state", 42)
        return cls(**xgb_params)

    if model_type == "neural_network":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPClassifier if task == "classification" else MLPRegressor
        return cls(**params)

def get_hyperparameter_docs(model_type: ModelType) -> dict[str, str]:
    """Return human-readable descriptions of each hyperparameter."""
    return HYPERPARAMETER_DOCS.get(model_type, {})


def list_model_types() -> list[ModelType]:
    return list(DEFAULTS.keys())
