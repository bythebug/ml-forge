from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from data.data_loader import dataset_statistics, load_dataset, missing_value_report
from db.models import Project
from db.session import get_db

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
