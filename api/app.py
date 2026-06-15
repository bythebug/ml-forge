from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data.data_loader import dataset_statistics, load_dataset, missing_value_report
from db.models import Base, FeatureSet, Project, TrainingRun, User
from db.session import engine, get_db
from evaluation.analysis import (
    confusion_matrix_analysis,
    error_examples,
    feature_importance_analysis,
    residual_analysis,
)
from evaluation.comparator import compare_runs, find_best_model
from evaluation.evaluator import bootstrap_confidence_interval, evaluate_model
from features.feature_builder import build_feature_set, validate_feature_definitions
from features.feature_selector import (
    backward_elimination,
    forward_selection,
    information_value,
    recursive_elimination,
    statistical_selection,
    variance_threshold,
)
from models.trainer import load_model, model_save_path, save_model, train_multiple
from tracking.mlflow_integration import MLflowTracker

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="ml-forge", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── GET /projects ────────────────────────────────────────────────────────────

@app.get("/projects")
def list_projects(db: Session = Depends(get_db)):
    """List all projects with run counts and best accuracy."""
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    result = []
    for p in projects:
        runs = db.query(TrainingRun).filter(TrainingRun.project_id == p.id).all()
        best_accuracy = None
        for r in runs:
            acc = (r.metrics or {}).get("accuracy")
            if acc is not None:
                best_accuracy = max(best_accuracy, acc) if best_accuracy else acc
        recent_model_types = list({r.model_type for r in runs[-3:]}) if runs else []
        result.append({
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "dataset_path": p.dataset_path,
            "run_count": len(runs),
            "best_accuracy": best_accuracy,
            "recent_model_types": recent_model_types,
            "created_at": p.created_at,
        })
    return result


# ─── POST /projects ───────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


@app.post("/projects", status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project under the default demo user (id=1)."""
    demo_user = db.get(User, 1)
    if not demo_user:
        demo_user = User(id=1, email="demo@ml-forge.local")
        db.add(demo_user)
        db.flush()

    project = Project(user_id=demo_user.id, name=body.name, description=body.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at,
    }


# ─── POST /projects/{project_id}/load_data ───────────────────────────────────

@app.post("/projects/{project_id}/load_data", status_code=status.HTTP_200_OK)
async def load_data(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a dataset file (CSV or Parquet) and attach it to a project."""
    project = _get_project_or_404(db, project_id)

    allowed_extensions = {".csv", ".parquet", ".pq"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Accepted: {allowed_extensions}",
        )

    dest = UPLOAD_DIR / f"project_{project_id}{suffix}"
    with dest.open("wb") as f:
        f.write(await file.read())

    df = load_dataset(str(dest))

    project.dataset_path = str(dest)
    db.commit()

    return {
        "project_id": project_id,
        "dataset_path": str(dest),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
    }


# ─── GET /projects/{project_id}/data_profile ─────────────────────────────────

@app.get("/projects/{project_id}/data_profile")
def data_profile(project_id: int, db: Session = Depends(get_db)):
    """Return full EDA statistics for a project's dataset."""
    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)
    return {"project_id": project_id, **dataset_statistics(df)}


# ─── GET /projects/{project_id}/missing_values ───────────────────────────────

@app.get("/projects/{project_id}/missing_values")
def missing_values(project_id: int, db: Session = Depends(get_db)):
    """Return per-column missing value counts and percentages for visualization."""
    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)
    return {
        "project_id": project_id,
        "total_rows": len(df),
        "columns": missing_value_report(df),
    }


# ─── Feature set request model ───────────────────────────────────────────────

class FeatureSetCreate(BaseModel):
    name: str
    features_spec: dict  # must match FeatureSet.features_list JSONB shape


# ─── POST /projects/{project_id}/feature_sets ────────────────────────────────

@app.post("/projects/{project_id}/feature_sets", status_code=status.HTTP_201_CREATED)
def create_feature_set(
    project_id: int,
    body: FeatureSetCreate,
    db: Session = Depends(get_db),
):
    """Create a named feature set from a feature spec and persist it."""
    _get_project_or_404(db, project_id)

    errors = validate_feature_definitions(body.features_spec)
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"validation_errors": errors},
        )

    feature_set = FeatureSet(
        project_id=project_id,
        name=body.name,
        features_list=body.features_spec,
    )
    db.add(feature_set)
    db.commit()
    db.refresh(feature_set)

    return {
        "id": feature_set.id,
        "project_id": project_id,
        "name": feature_set.name,
        "feature_count": len(body.features_spec.get("features", [])),
        "created_at": feature_set.created_at,
    }


# ─── GET /projects/{project_id}/feature_sets ─────────────────────────────────

@app.get("/projects/{project_id}/feature_sets")
def list_feature_sets(project_id: int, db: Session = Depends(get_db)):
    """List all feature sets for a project."""
    _get_project_or_404(db, project_id)
    feature_sets = db.query(FeatureSet).filter(FeatureSet.project_id == project_id).all()
    return [
        {
            "id": fs.id,
            "name": fs.name,
            "feature_count": len(fs.features_list.get("features", [])),
            "created_at": fs.created_at,
        }
        for fs in feature_sets
    ]


# ─── GET /projects/{project_id}/feature_sets/{feature_set_id}/preview ────────

@app.get("/projects/{project_id}/feature_sets/{feature_set_id}/preview")
def preview_feature_set(
    project_id: int,
    feature_set_id: int,
    rows: int = 5,
    db: Session = Depends(get_db),
):
    """Apply the feature spec to the project dataset and return the first N rows."""
    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)

    feature_set = db.get(FeatureSet, feature_set_id)
    if not feature_set or feature_set.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature set {feature_set_id} not found in project {project_id}.",
        )

    augmented = build_feature_set(df, feature_set.features_list)
    preview = augmented.head(rows)

    return {
        "feature_set_id": feature_set_id,
        "feature_set_name": feature_set.name,
        "original_columns": df.columns.tolist(),
        "engineered_columns": [c for c in augmented.columns if c not in df.columns],
        "preview": preview.where(preview.notna(), other=None).to_dict(orient="records"),
    }


# ─── Feature selection request model ─────────────────────────────────────────

class SelectFeaturesRequest(BaseModel):
    target: str
    method: Literal[
        "correlation", "spearman", "variance_threshold", "forward", "backward", "iv"
    ] = "correlation"
    n_features: int = 10
    threshold: float = 0.05


# ─── GET /projects/{project_id}/feature_importance ───────────────────────────

@app.get("/projects/{project_id}/feature_importance")
def feature_importance(
    project_id: int,
    target: str,
    method: str = "correlation",
    db: Session = Depends(get_db),
):
    """Return each feature's correlation (or mutual info) score with the target column."""
    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)

    if target not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{target}' not found in dataset.",
        )

    result = statistical_selection(df, target=target, method=method, threshold=0.0)
    return {
        "project_id": project_id,
        "target": target,
        "method": method,
        "importance_scores": result["scores"],
    }


# ─── POST /projects/{project_id}/select_features ─────────────────────────────

@app.post("/projects/{project_id}/select_features")
def select_features(
    project_id: int,
    body: SelectFeaturesRequest,
    db: Session = Depends(get_db),
):
    """Run a feature selection method and return the selected column names."""
    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)

    if body.target not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{body.target}' not found.",
        )

    method = body.method
    try:
        if method in ("correlation", "spearman", "mutual_info"):
            result = statistical_selection(
                df, target=body.target, method=method, threshold=body.threshold
            )
        elif method == "variance_threshold":
            result = variance_threshold(df, threshold=body.threshold)
        elif method == "forward":
            result = forward_selection(df, target=body.target, max_features=body.n_features)
        elif method == "backward":
            result = backward_elimination(df, target=body.target, max_features=body.n_features)
        elif method == "iv":
            iv_scores = information_value(df, target_col=body.target)
            selected = list(iv_scores.keys())[: body.n_features]
            result = {"method": "iv", "selected": selected, "scores": iv_scores}
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown selection method '{method}'.",
            )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return {"project_id": project_id, **result}


# ─── GET /projects/{project_id}/feature_stats ────────────────────────────────

@app.get("/projects/{project_id}/feature_stats")
def feature_stats(project_id: int, db: Session = Depends(get_db)):
    """Return correlation matrix and per-column distribution stats."""
    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)

    numeric = df.select_dtypes(include=["number"])

    corr_matrix = numeric.corr().round(4).where(
        numeric.corr().notna(), other=None
    ).to_dict()

    distribution = {
        col: {
            "mean":     round(float(numeric[col].mean()), 4),
            "std":      round(float(numeric[col].std()), 4),
            "skewness": round(float(numeric[col].skew()), 4),
            "kurtosis": round(float(numeric[col].kurtosis()), 4),
            "min":      round(float(numeric[col].min()), 4),
            "max":      round(float(numeric[col].max()), 4),
        }
        for col in numeric.columns
    }

    return {
        "project_id": project_id,
        "numeric_columns": numeric.columns.tolist(),
        "correlation_matrix": corr_matrix,
        "distribution_stats": distribution,
    }


# ─── Training run request model ──────────────────────────────────────────────

class ModelConfig(BaseModel):
    type: str
    hyperparams: dict = {}


class TrainRunRequest(BaseModel):
    feature_set_id: int
    target_col: str
    models: list[ModelConfig]
    task: Literal["classification", "regression"] = "classification"
    test_size: float = 0.2
    random_state: int = 42


# ─── POST /projects/{project_id}/train_run ────────────────────────────────────

@app.post("/projects/{project_id}/train_run", status_code=status.HTTP_201_CREATED)
def train_run(
    project_id: int,
    body: TrainRunRequest,
    db: Session = Depends(get_db),
):
    """Train one or more models on the project dataset and persist results."""
    from sklearn.model_selection import train_test_split
    from features.normalizer import Scaler

    project = _get_project_or_404(db, project_id)
    df = _load_project_dataset(project)

    feature_set = db.get(FeatureSet, body.feature_set_id)
    if not feature_set or feature_set.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature set {body.feature_set_id} not found in project {project_id}.",
        )

    if body.target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{body.target_col}' not found in dataset.",
        )

    augmented = build_feature_set(df, feature_set.features_list)
    feature_cols = [c for c in augmented.select_dtypes(include=["number"]).columns
                    if c != body.target_col]

    X = augmented[feature_cols].fillna(augmented[feature_cols].mean())
    y = augmented[body.target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=body.test_size, random_state=body.random_state
    )

    numeric_cols = X_train.columns.tolist()
    scaler = Scaler("standard").fit(X_train, numeric_cols)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # get-or-create MLflow experiment for this project (silently skip if unavailable)
    mlflow_experiment_id: Optional[str] = None
    try:
        tracker = MLflowTracker()
        mlflow_experiment_id = tracker.get_or_create_experiment(
            project_id=project_id,
            project_name=project.name,
        )
    except Exception:
        pass

    configs = [{"type": m.type, "hyperparams": m.hyperparams} for m in body.models]
    results = train_multiple(
        configs, X_train_s, y_train, X_val_s, y_val,
        task=body.task,
        mlflow_experiment_id=mlflow_experiment_id,
        mlflow_feature_set_name=feature_set.name,
        mlflow_project_id=project_id,
    )

    run_records = []
    for result in results:
        run = TrainingRun(
            project_id=project_id,
            feature_set_id=body.feature_set_id,
            model_type=result.model_type,
            metrics=result.to_metrics_dict(),
        )
        db.add(run)
        db.flush()  # get run.id before commit

        model_path = model_save_path(project_id, run.id, result.model_type)
        save_model(result.model, model_path)
        run.metrics = {**run.metrics, "model_path": str(model_path)}
        run_records.append(run)

    db.commit()

    return {
        "project_id": project_id,
        "feature_set_id": body.feature_set_id,
        "runs": [
            {
                "run_id": r.id,
                "model_type": r.model_type,
                "metrics": r.metrics,
                "created_at": r.created_at,
            }
            for r in run_records
        ],
    }


# ─── GET /projects/{project_id}/runs ─────────────────────────────────────────

@app.get("/projects/{project_id}/runs")
def list_runs(project_id: int, db: Session = Depends(get_db)):
    """List all training runs for a project, sorted by creation time descending."""
    _get_project_or_404(db, project_id)
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.created_at.desc())
        .all()
    )
    return [
        {
            "run_id": r.id,
            "model_type": r.model_type,
            "feature_set_id": r.feature_set_id,
            "status": (r.metrics or {}).get("status", "unknown"),
            "primary_metric": _primary_metric(r.metrics),
            "training_time_s": (r.metrics or {}).get("training_time_s"),
            "created_at": r.created_at,
        }
        for r in runs
    ]


# ─── GET /projects/{project_id}/runs/{run_id}/progress ───────────────────────

@app.get("/projects/{project_id}/runs/{run_id}/progress")
def run_progress(project_id: int, run_id: int, db: Session = Depends(get_db)):
    """Return the status and full metrics for a specific training run."""
    _get_project_or_404(db, project_id)
    run = db.get(TrainingRun, run_id)

    if not run or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}.",
        )

    return {
        "run_id": run_id,
        "project_id": project_id,
        "model_type": run.model_type,
        "feature_set_id": run.feature_set_id,
        "status": (run.metrics or {}).get("status", "unknown"),
        "metrics": run.metrics,
        "created_at": run.created_at,
    }


def _primary_metric(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    for key in ("accuracy", "roc_auc", "f1", "r2"):
        if key in metrics:
            return {key: metrics[key]}
    return None


# ─── GET /projects/{project_id}/comparison ───────────────────────────────────

@app.get("/projects/{project_id}/comparison")
def comparison(
    project_id: int,
    metric: str = "accuracy",
    db: Session = Depends(get_db),
):
    """Compare all training runs for a project, ranked by `metric`."""
    _get_project_or_404(db, project_id)
    runs = db.query(TrainingRun).filter(TrainingRun.project_id == project_id).all()
    if not runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training runs found for this project.",
        )

    run_dicts = [
        {
            "run_id": r.id,
            "model_type": r.model_type,
            "feature_set_id": r.feature_set_id,
            "metrics": r.metrics or {},
            "created_at": str(r.created_at),
        }
        for r in runs
    ]
    return {"project_id": project_id, **compare_runs(run_dicts, primary_metric=metric)}


# ─── GET /projects/{project_id}/best_model ───────────────────────────────────

@app.get("/projects/{project_id}/best_model")
def best_model(
    project_id: int,
    metric: str = "accuracy",
    db: Session = Depends(get_db),
):
    """Return the best-performing run with a recommendation and confidence interval."""
    _get_project_or_404(db, project_id)
    runs = db.query(TrainingRun).filter(TrainingRun.project_id == project_id).all()
    if not runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No training runs found.",
        )

    run_dicts = [
        {"run_id": r.id, "model_type": r.model_type, "metrics": r.metrics or {}}
        for r in runs
    ]
    result = find_best_model(run_dicts, primary_metric=metric)

    # add confidence interval for accuracy/F1 metrics
    best = result["best_run"]
    n_test = _infer_n_test_from_run(best)
    metric_val = (best.get("metrics") or {}).get(metric)
    if metric_val is not None and n_test:
        try:
            lo, hi = bootstrap_confidence_interval(metric_val, n_test)
            result["confidence_interval_95"] = {"lower": lo, "upper": hi}
        except Exception:
            pass

    return {"project_id": project_id, **result}


# ─── GET /projects/{project_id}/runs/{run_id}/analysis ───────────────────────

@app.get("/projects/{project_id}/runs/{run_id}/analysis")
def run_analysis(
    project_id: int,
    run_id: int,
    target_col: str,
    task: str = "classification",
    db: Session = Depends(get_db),
):
    """Detailed error analysis for a specific training run."""
    from sklearn.model_selection import train_test_split
    from features.normalizer import Scaler

    project = _get_project_or_404(db, project_id)
    run = db.get(TrainingRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found in project {project_id}.",
        )

    model_path = (run.metrics or {}).get("model_path")
    if not model_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model file path not recorded for this run.",
        )

    model = load_model(model_path)
    df = _load_project_dataset(project)

    feature_set = db.get(FeatureSet, run.feature_set_id)
    augmented = build_feature_set(df, feature_set.features_list)
    feature_cols = [c for c in augmented.select_dtypes(include=["number"]).columns
                    if c != target_col]

    X = augmented[feature_cols].fillna(augmented[feature_cols].mean())
    y = augmented[target_col]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = Scaler("standard").fit(X, feature_cols)
    X_test_s = scaler.transform(X_test)

    y_pred = model.predict(X_test_s)
    y_true = y_test.values

    response: dict = {
        "run_id": run_id,
        "model_type": run.model_type,
        "task": task,
        "n_test_samples": int(len(y_true)),
    }

    if task == "classification":
        response["confusion_matrix_analysis"] = confusion_matrix_analysis(y_true, y_pred)
        response["error_examples"] = error_examples(X_test, y_true, y_pred, n=10)
    else:
        response["residual_analysis"] = residual_analysis(y_true, y_pred)
        response["error_examples"] = error_examples(X_test, y_true, y_pred, n=10, task="regression")

    response["feature_importance"] = feature_importance_analysis(model, feature_cols)

    return response


# ─── helpers ─────────────────────────────────────────────────────────────────

# ─── GET /projects/{project_id}/experiments ──────────────────────────────────

@app.get("/projects/{project_id}/experiments")
def get_experiments(project_id: int, db: Session = Depends(get_db)):
    """Return the MLflow experiment for this project and all logged runs."""
    project = _get_project_or_404(db, project_id)

    try:
        tracker = MLflowTracker()
        experiment_id = tracker.get_or_create_experiment(project_id, project.name)
        experiment = tracker.get_experiment(experiment_id)
        runs = tracker.get_runs(experiment_id)
        best = tracker.get_best_run(experiment_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow unavailable: {exc}. Ensure MLFLOW_TRACKING_URI is set and the server is running.",
        )

    return {
        "project_id": project_id,
        "experiment": experiment,
        "mlflow_ui_url": tracker.ui_url(experiment_id=experiment_id),
        "total_runs": len(runs),
        "runs": runs,
        "best_run": best,
    }


# ─── GET /projects/{project_id}/experiments/best ─────────────────────────────

@app.get("/projects/{project_id}/experiments/best")
def best_mlflow_run(
    project_id: int,
    metric: str = "accuracy",
    db: Session = Depends(get_db),
):
    """Return the best MLflow run for a metric, with a direct UI link."""
    project = _get_project_or_404(db, project_id)

    try:
        tracker = MLflowTracker()
        experiment_id = tracker.get_or_create_experiment(project_id, project.name)
        best = tracker.get_best_run(experiment_id, metric=metric)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLflow unavailable: {exc}",
        )

    if not best:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No runs logged for project {project_id}.",
        )

    return {
        "project_id": project_id,
        "metric": metric,
        "best_run": best,
        "mlflow_ui_url": tracker.ui_url(
            experiment_id=experiment_id,
            run_id=best["run_id"],
        ),
    }


def _infer_n_test_from_run(run: dict) -> Optional[int]:
    cm = (run.get("metrics") or {}).get("confusion_matrix")
    if cm:
        return int(sum(sum(row) for row in cm))
    return None


def _get_project_or_404(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found.",
        )
    return project


def _load_project_dataset(project: Project):
    if not project.dataset_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No dataset loaded for this project. POST /load_data first.",
        )
    try:
        return load_dataset(project.dataset_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file not found at: {project.dataset_path}",
        )
