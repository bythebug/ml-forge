from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy


_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}
_SQL_SCHEMES = ("postgresql://", "postgresql+psycopg2://", "sqlite://", "mysql://")


def load_dataset(
    dataset_path: str,
    sql_query: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a dataset from CSV, Parquet, or SQL source into a DataFrame."""
    if _is_sql_source(dataset_path):
        return _load_from_sql(dataset_path, sql_query, **kwargs)

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(
            f"Unsupported format '{suffix}'. Use .csv, .parquet, or a SQL connection string."
        )


def dataset_statistics(df: pd.DataFrame) -> dict:
    """Return shape, dtypes, missing values, and per-column descriptive stats."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    numeric_stats = (
        df[numeric_cols].describe().round(4).to_dict() if numeric_cols else {}
    )

    categorical_stats = {
        col: {
            "unique": int(df[col].nunique()),
            "top": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            "freq": int(df[col].value_counts().iloc[0]) if df[col].notna().any() else 0,
        }
        for col in categorical_cols
    }

    return {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": missing_counts.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
    }


def missing_value_report(df: pd.DataFrame) -> list[dict]:
    """Per-column missing value summary sorted by missing count descending."""
    records = []
    for col in df.columns:
        count = int(df[col].isnull().sum())
        records.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": count,
                "missing_pct": round(count / len(df) * 100, 2),
            }
        )
    return sorted(records, key=lambda r: r["missing_count"], reverse=True)


# ─── internal helpers ────────────────────────────────────────────────────────

def _is_sql_source(path: str) -> bool:
    return any(path.startswith(scheme) for scheme in _SQL_SCHEMES)


def _load_from_sql(connection_string: str, query: Optional[str], **kwargs) -> pd.DataFrame:
    if not query:
        raise ValueError("sql_query is required when dataset_path is a SQL connection string.")
    engine = sqlalchemy.create_engine(connection_string)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, **kwargs)
