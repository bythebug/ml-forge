from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data.data_loader import dataset_statistics, load_dataset, missing_value_report
from db.models import FeatureSet, Project
from db.session import get_db
from features.feature_builder import build_feature_set, validate_feature_definitions
from features.feature_selector import (
    backward_elimination,
    forward_selection,
    information_value,
    recursive_elimination,
    statistical_selection,
    variance_threshold,
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ml-forge", version="0.1.0")


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


# ─── helpers ─────────────────────────────────────────────────────────────────

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
