"""
MLflow experiment tracking.

One experiment per project (named "project_{id}_{name}").
One run per (feature_set × model_type × hyperparams) combination.

Usage:
    from tracking.mlflow_integration import MLflowTracker
    tracker = MLflowTracker()
    experiment_id = tracker.get_or_create_experiment(project_id=1, project_name="churn")
    run_id = tracker.log_run(result, experiment_id, feature_set_name="v1", ...)
"""

import os
from typing import Any, Optional

import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlflow_artifacts")

# Metrics that MLflow should not try to log (non-scalar)
_SKIP_METRIC_KEYS = {"confusion_matrix", "per_class", "residuals", "woe", "status",
                     "hyperparams", "model_path", "mlflow_run_id", "task"}


class MLflowTracker:
    """Thin wrapper around the MLflow client for ml-forge."""

    def __init__(self, tracking_uri: Optional[str] = None) -> None:
        if tracking_uri:
            # Explicit URI — set it globally (used by tests and explicit config)
            self.tracking_uri = tracking_uri
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Re-read env var at construction time so monkeypatching works in tests
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
            mlflow.set_tracking_uri(self.tracking_uri)
        self._client = mlflow.MlflowClient()

    # ── experiments ───────────────────────────────────────────────────────────

    def get_or_create_experiment(
        self,
        project_id: int,
        project_name: str,
    ) -> str:
        """Return existing experiment_id or create a new one."""
        name = self._experiment_name(project_id, project_name)
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        return self._client.create_experiment(
            name,
            tags={"project_id": str(project_id)},
        )

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        """Return experiment metadata or None if it doesn't exist."""
        try:
            exp = self._client.get_experiment(experiment_id)
            return {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
                "tags": dict(exp.tags),
            }
        except Exception:
            return None

    # ── runs ──────────────────────────────────────────────────────────────────

    def log_run(
        self,
        result: Any,                  # TrainResult from trainer.py
        experiment_id: str,
        feature_set_name: str,
        project_id: int,
        db_run_id: Optional[int] = None,
        log_model_artifact: bool = True,
    ) -> str:
        """
        Log a TrainResult to MLflow.
        Returns the MLflow run_id string.

        Logs:
          - Tags: project_id, model_type, feature_set, db_run_id
          - Params: all hyperparameters
          - Metrics: all scalar metrics (accuracy, f1, roc_auc, training_time_s, …)
          - Artifact: serialised sklearn model (if log_model_artifact=True)
        """
        run_name = f"{result.model_type}__{feature_set_name}"

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        ) as run:
            # ── tags ──
            mlflow.set_tags({
                "project_id":    str(project_id),
                "model_type":    result.model_type,
                "feature_set":   feature_set_name,
                "task":          result.task,
                **({"db_run_id": str(db_run_id)} if db_run_id else {}),
            })

            # ── hyperparameters ──
            safe_params = {
                k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                for k, v in result.hyperparams.items()
            }
            mlflow.log_params(safe_params)

            # ── metrics (scalar only) ──
            scalar_metrics = {
                k: float(v)
                for k, v in result.metrics.items()
                if k not in _SKIP_METRIC_KEYS
                and isinstance(v, (int, float))
                and not isinstance(v, bool)
            }
            scalar_metrics["training_time_s"] = result.training_time_s
            mlflow.log_metrics(scalar_metrics)

            # ── model artifact ──
            if log_model_artifact:
                try:
                    mlflow.sklearn.log_model(result.model, artifact_path="model")
                except Exception:
                    pass  # non-sklearn models fall back silently

            return run.info.run_id

    # ── querying ──────────────────────────────────────────────────────────────

    def get_runs(
        self,
        experiment_id: str,
        order_by: str = "metrics.accuracy DESC",
        max_results: int = 100,
    ) -> list[dict]:
        """Return all runs for an experiment, ordered by metric."""
        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[order_by],
            max_results=max_results,
        )
        return [self._run_to_dict(r) for r in runs]

    def get_best_run(
        self,
        experiment_id: str,
        metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Optional[dict]:
        """Return the best run for a given metric."""
        direction = "DESC" if higher_is_better else "ASC"
        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric} {direction}"],
            max_results=1,
        )
        return self._run_to_dict(runs[0]) if runs else None

    def get_run(self, run_id: str) -> Optional[dict]:
        """Return a single run by its MLflow run_id."""
        try:
            run = self._client.get_run(run_id)
            return self._run_to_dict(run)
        except Exception:
            return None

    # ── UI link ───────────────────────────────────────────────────────────────

    def ui_url(self, experiment_id: Optional[str] = None, run_id: Optional[str] = None) -> str:
        base = self.tracking_uri.rstrip("/")
        if run_id:
            return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"
        if experiment_id:
            return f"{base}/#/experiments/{experiment_id}"
        return f"{base}/#/experiments"

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _experiment_name(project_id: int, project_name: str) -> str:
        safe = project_name.replace(" ", "_").lower()
        return f"project_{project_id}_{safe}"

    @staticmethod
    def _run_to_dict(run: Any) -> dict:
        return {
            "run_id":     run.info.run_id,
            "run_name":   run.info.run_name,
            "status":     run.info.status,
            "start_time": run.info.start_time,
            "end_time":   run.info.end_time,
            "tags":       dict(run.data.tags),
            "params":     dict(run.data.params),
            "metrics":    dict(run.data.metrics),
        }
