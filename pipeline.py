#!/usr/bin/env python3
"""
ml-forge end-to-end pipeline.

Orchestrates the full ML workflow:
  load → preprocess → engineer features → select → scale → train → compare → report

Usage:
    # Programmatic
    from pipeline import run_pipeline, PipelineConfig
    config = PipelineConfig(dataset_path="data/churn.csv", target_col="churn")
    result = run_pipeline(config)
    print(result.report)

    # CLI
    python pipeline.py --dataset data/churn.csv --target churn --models random_forest,xgboost
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.data_loader import dataset_statistics, load_dataset
from data.preprocessor import handle_missing_values
from evaluation.comparator import compare_runs, find_best_model
from features.feature_builder import build_feature_set
from features.feature_selector import statistical_selection, variance_threshold
from features.normalizer import Scaler
from models.trainer import TrainResult, train_multiple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml-forge")

# ── default feature spec (raw passthrough of all numeric columns) ─────────────

_DEFAULT_FEATURE_SPEC = {"features": []}   # empty = auto-detect numeric columns

# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    dataset_path: str
    target_col: str
    task: str                      = "classification"
    feature_spec: dict             = field(default_factory=dict)
    models: list[dict]             = field(default_factory=list)
    missing_strategy: str          = "mean"
    variance_threshold: float      = 0.01
    correlation_threshold: float   = 0.02
    test_size: float               = 0.2
    random_state: int              = 42
    mlflow_project_id: Optional[int] = None
    mlflow_experiment_name: str    = "ml-forge-pipeline"

    def __post_init__(self):
        if not self.models:
            self.models = [
                {"type": "logistic_regression", "hyperparams": {}},
                {"type": "random_forest",       "hyperparams": {"n_estimators": 100}},
                {"type": "xgboost",             "hyperparams": {"n_estimators": 100, "learning_rate": 0.1}},
            ]


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class StepTiming:
    step: str
    duration_s: float


@dataclass
class PipelineResult:
    config: PipelineConfig
    data_stats: dict
    selected_features: list[str]
    train_results: list[TrainResult]
    comparison: dict
    timings: list[StepTiming]
    report: str


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full ml-forge pipeline and return a structured result."""
    timings: list[StepTiming] = []

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    log.info("━" * 60)
    log.info("STEP 1 / 7  Load dataset")
    t0 = time.perf_counter()

    df = load_dataset(config.dataset_path)
    stats = dataset_statistics(df)

    timings.append(StepTiming("load", time.perf_counter() - t0))
    log.info("  rows=%d  cols=%d  missing=%d",
             stats["shape"]["rows"], stats["shape"]["columns"],
             sum(stats["missing"].values()))

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    log.info("STEP 2 / 7  Preprocess (missing values)")
    t0 = time.perf_counter()

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != config.target_col]
    df = handle_missing_values(df, strategy=config.missing_strategy, columns=numeric_cols)

    timings.append(StepTiming("preprocess", time.perf_counter() - t0))
    log.info("  strategy=%s  remaining_nulls=%d",
             config.missing_strategy, df[numeric_cols].isnull().sum().sum())

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    log.info("STEP 3 / 7  Feature engineering")
    t0 = time.perf_counter()

    if config.feature_spec.get("features"):
        df = build_feature_set(df, config.feature_spec)
        log.info("  applied custom feature spec  new_cols=%d",
                 len(config.feature_spec["features"]))
    else:
        log.info("  no custom spec — using raw numeric columns")

    timings.append(StepTiming("feature_engineering", time.perf_counter() - t0))

    # ── Step 4: Feature selection ─────────────────────────────────────────────
    log.info("STEP 4 / 7  Feature selection")
    t0 = time.perf_counter()

    # low-variance filter
    vt = variance_threshold(df, threshold=config.variance_threshold,
                            exclude=[config.target_col])
    df = df[vt["selected"] + [config.target_col]]

    # correlation filter
    corr = statistical_selection(df, target=config.target_col,
                                 threshold=config.correlation_threshold)
    selected_features = corr["selected"]

    if not selected_features:
        selected_features = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c != config.target_col]

    timings.append(StepTiming("feature_selection", time.perf_counter() - t0))
    log.info("  after variance filter=%d  after correlation filter=%d",
             len(vt["selected"]), len(selected_features))

    # ── Step 5: Train/test split + scaling ────────────────────────────────────
    log.info("STEP 5 / 7  Split and scale")
    t0 = time.perf_counter()

    X = df[selected_features].fillna(df[selected_features].mean())
    y = df[config.target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if config.task == "classification" else None,
    )

    scaler = Scaler("standard").fit(X_train, selected_features)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)

    timings.append(StepTiming("split_scale", time.perf_counter() - t0))
    log.info("  train=%d  val=%d  features=%d", len(X_train), len(X_val), len(selected_features))

    # ── Step 6: Train models ──────────────────────────────────────────────────
    log.info("STEP 6 / 7  Train %d model(s)", len(config.models))
    t0 = time.perf_counter()

    mlflow_experiment_id: Optional[str] = None
    if config.mlflow_project_id is not None:
        try:
            from tracking.mlflow_integration import MLflowTracker
            tracker = MLflowTracker()
            mlflow_experiment_id = tracker.get_or_create_experiment(
                config.mlflow_project_id, config.mlflow_experiment_name
            )
        except Exception:
            pass

    results = train_multiple(
        config.models,
        X_train_s, y_train,
        X_val_s, y_val,
        task=config.task,
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_feature_set_name=config.mlflow_experiment_name,
        mlflow_project_id=config.mlflow_project_id,
    )

    for r in results:
        primary = r.metrics.get("accuracy") or r.metrics.get("r2")
        log.info("  %-22s  metric=%.4f  time=%.2fs",
                 r.model_type, primary or 0, r.training_time_s)

    timings.append(StepTiming("training", time.perf_counter() - t0))

    # ── Step 7: Compare and report ────────────────────────────────────────────
    log.info("STEP 7 / 7  Compare and rank")

    primary_metric = "accuracy" if config.task == "classification" else "r2"
    run_dicts = [
        {"run_id": i, "model_type": r.model_type, "metrics": r.metrics}
        for i, r in enumerate(results)
    ]
    comparison = compare_runs(run_dicts, primary_metric=primary_metric)
    report = _build_report(config, stats, selected_features, results, comparison,
                           timings, primary_metric)

    log.info("━" * 60)
    log.info("WINNER  %s", comparison["winner"].get("model_type", "N/A"))
    log.info("━" * 60)

    return PipelineResult(
        config=config,
        data_stats=stats,
        selected_features=selected_features,
        train_results=results,
        comparison=comparison,
        timings=timings,
        report=report,
    )


# ── report builder ────────────────────────────────────────────────────────────

def _build_report(
    config: PipelineConfig,
    stats: dict,
    features: list[str],
    results: list[TrainResult],
    comparison: dict,
    timings: list[StepTiming],
    primary_metric: str,
) -> str:
    sep = "=" * 62
    lines = [
        sep,
        "  ml-forge Pipeline Report",
        sep,
        f"  Dataset       : {config.dataset_path}",
        f"  Target        : {config.target_col}",
        f"  Task          : {config.task}",
        f"  Rows / Cols   : {stats['shape']['rows']} / {stats['shape']['columns']}",
        f"  Features used : {len(features)}",
        "",
        "  Feature selection",
        "  " + "-" * 58,
    ]
    for f in features[:10]:
        lines.append(f"    {f}")
    if len(features) > 10:
        lines.append(f"    … and {len(features) - 10} more")

    lines += [
        "",
        "  Model Rankings",
        "  " + "-" * 58,
        f"  {'Model':<24} {primary_metric.upper():<10} F1         Time(s)",
    ]
    for r in results:
        acc  = r.metrics.get(primary_metric, 0)
        f1   = r.metrics.get("f1", r.metrics.get("mae", 0))
        lines.append(f"  {r.model_type:<24} {acc:<10.4f} {f1:<10.4f} {r.training_time_s:.2f}")

    winner = comparison.get("winner", {})
    lines += [
        "",
        f"  Winner : {winner.get('model_type', 'N/A')}",
        f"  Margin : {winner.get('margin_over_second', 0):.4f} over second place",
        "",
        "  Step Timings",
        "  " + "-" * 58,
    ]
    total = sum(t.duration_s for t in timings)
    for t in timings:
        lines.append(f"  {t.step:<22} {t.duration_s:.3f}s")
    lines += [f"  {'TOTAL':<22} {total:.3f}s", sep]

    return "\n".join(lines)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ml-forge end-to-end pipeline")
    p.add_argument("--dataset",      required=True,  help="Path to CSV or Parquet file")
    p.add_argument("--target",       required=True,  help="Target column name")
    p.add_argument("--task",         default="classification",
                   choices=["classification", "regression"])
    p.add_argument("--models",       default="logistic_regression,random_forest,xgboost",
                   help="Comma-separated model types")
    p.add_argument("--test-size",    type=float, default=0.2)
    p.add_argument("--random-state", type=int,   default=42)
    p.add_argument("--missing",      default="mean",
                   choices=["mean", "median", "mode", "drop", "forward_fill", "constant"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = PipelineConfig(
        dataset_path=args.dataset,
        target_col=args.target,
        task=args.task,
        models=[{"type": m.strip(), "hyperparams": {}} for m in args.models.split(",")],
        test_size=args.test_size,
        random_state=args.random_state,
        missing_strategy=args.missing,
    )
    result = run_pipeline(cfg)
    print(result.report)
