"""
Training orchestration — fits models, evaluates them, and handles persistence.

    result = train_model("random_forest", X_train, y_train, X_val, y_val)
    print(result.metrics)
    save_model(result.model, "models/saved/rf.joblib")
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from models.model_definitions import ModelType, Task, build_model

SAVED_DIR = Path("models/saved")
SAVED_DIR.mkdir(parents=True, exist_ok=True)


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    model_type: ModelType
    task: Task
    model: Any
    hyperparams: dict
    metrics: dict
    training_time_s: float
    model_path: Optional[str] = None

    def to_metrics_dict(self) -> dict:
        """Serialisable dict for storage in TrainingRun.metrics JSONB."""
        return {
            "status": "completed",
            "task": self.task,
            "training_time_s": round(self.training_time_s, 3),
            "hyperparams": self.hyperparams,
            "model_path": self.model_path,
            **self.metrics,
        }


# ── core training function ────────────────────────────────────────────────────

def train_model(
    model_type: ModelType,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: Optional[np.ndarray | pd.DataFrame] = None,
    y_val: Optional[np.ndarray | pd.Series] = None,
    hyperparams: Optional[dict] = None,
    task: Task = "classification",
) -> TrainResult:
    """
    Build, fit, and evaluate one model. Returns a TrainResult with metrics.

    If X_val / y_val are provided, reports held-out metrics.
    Otherwise, reports training-set metrics only (use with caution — optimistic).
    """
    model = build_model(model_type, hyperparams, task)
    hp = {**_get_defaults(model_type), **(hyperparams or {})}

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    X_eval = X_val if X_val is not None else X_train
    y_eval = y_val if y_val is not None else y_train
    metrics = _evaluate(model, X_eval, y_eval, task)

    return TrainResult(
        model_type=model_type,
        task=task,
        model=model,
        hyperparams=hp,
        metrics=metrics,
        training_time_s=elapsed,
    )


# ── multi-model training ──────────────────────────────────────────────────────

def train_multiple(
    configs: list[dict],
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: Optional[np.ndarray | pd.DataFrame] = None,
    y_val: Optional[np.ndarray | pd.Series] = None,
    task: Task = "classification",
) -> list[TrainResult]:
    """
    Train each model in `configs` and return results sorted by primary metric.
    Config format: [{"type": "random_forest", "hyperparams": {...}}, ...]
    """
    results = []
    for cfg in configs:
        result = train_model(
            model_type=cfg["type"],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparams=cfg.get("hyperparams"),
            task=task,
        )
        results.append(result)

    sort_key = "accuracy" if task == "classification" else "r2"
    return sorted(results, key=lambda r: r.metrics.get(sort_key, 0), reverse=True)


# ── model persistence ─────────────────────────────────────────────────────────

def save_model(model: Any, path: str | Path) -> Path:
    """Serialise model to disk using joblib. Returns the resolved path."""
    import joblib
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_model(path: str | Path) -> Any:
    """Deserialise a model from a joblib file."""
    import joblib
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(p)


def model_save_path(project_id: int, run_id: int, model_type: str) -> Path:
    return SAVED_DIR / f"project_{project_id}_run_{run_id}_{model_type}.joblib"


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model: Any, X: Any, y: Any, task: Task) -> dict:
    if task == "classification":
        return _classification_metrics(model, X, y)
    return _regression_metrics(model, X, y)


def _classification_metrics(model: Any, X: Any, y: Any) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X)
    n_classes = len(np.unique(y))
    avg = "binary" if n_classes == 2 else "weighted"

    metrics: dict[str, Any] = {
        "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        "f1":        round(float(f1_score(y, y_pred, average=avg, zero_division=0)), 4),
        "precision": round(float(precision_score(y, y_pred, average=avg, zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, average=avg, zero_division=0)), 4),
    }

    if hasattr(model, "predict_proba") and n_classes == 2:
        proba = model.predict_proba(X)[:, 1]
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y, proba)), 4)
        except ValueError:
            pass  # can't compute AUC if only one class in eval set

    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def _regression_metrics(model: Any, X: Any, y: Any) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X)
    return {
        "r2":   round(float(r2_score(y, y_pred)), 4),
        "mae":  round(float(mean_absolute_error(y, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
    }


# ── private helpers ───────────────────────────────────────────────────────────

def _get_defaults(model_type: ModelType) -> dict:
    from models.model_definitions import DEFAULTS
    return dict(DEFAULTS.get(model_type, {}))
