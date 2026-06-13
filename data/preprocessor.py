from typing import Literal, Optional, Union

import numpy as np
import pandas as pd


MissingStrategy = Literal["drop", "mean", "median", "mode", "forward_fill", "backward_fill", "constant"]
OutlierMethod = Literal["iqr", "zscore"]
EncodingMethod = Literal["label", "onehot"]
NormalizationMethod = Literal["standard", "minmax", "robust"]


# ─── Missing values ──────────────────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame,
    strategy: MissingStrategy = "mean",
    columns: Optional[list[str]] = None,
    fill_value: Optional[Union[int, float, str]] = None,
) -> pd.DataFrame:
    """Fill or drop missing values using the given strategy."""
    df = df.copy()
    target_cols = columns or df.columns.tolist()

    if strategy == "drop":
        return df.dropna(subset=target_cols)

    for col in target_cols:
        if df[col].isnull().sum() == 0:
            continue

        if strategy == "mean":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        elif strategy == "forward_fill":
            df[col] = df[col].ffill()
        elif strategy == "backward_fill":
            df[col] = df[col].bfill()
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'.")
            df[col] = df[col].fillna(fill_value)

    return df


# ─── Outlier detection ───────────────────────────────────────────────────────

def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: OutlierMethod = "iqr",
) -> dict:
    """Return outlier bounds, count, and index positions for a numeric column."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for outlier detection.")

    series = df[column].dropna()

    if method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    elif method == "zscore":
        mean, std = series.mean(), series.std()
        lower, upper = mean - 3 * std, mean + 3 * std
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

    mask = (df[column] < lower) | (df[column] > upper)
    return {
        "column": column,
        "method": method,
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(float(mask.mean() * 100), 2),
        "outlier_indices": df.index[mask].tolist(),
    }


def remove_outliers(df: pd.DataFrame, column: str, method: OutlierMethod = "iqr") -> pd.DataFrame:
    """Drop rows identified as outliers in the given column."""
    result = detect_outliers(df, column, method)
    return df.drop(index=result["outlier_indices"]).reset_index(drop=True)


# ─── Categorical encoding ────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
    method: EncodingMethod = "label",
) -> pd.DataFrame:
    """Encode categorical columns using label or one-hot encoding."""
    df = df.copy()

    if method == "label":
        for col in columns:
            df[col] = df[col].astype("category").cat.codes.where(df[col].notna(), other=-1)
    elif method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=False)
    else:
        raise ValueError(f"Unknown encoding method '{method}'. Use 'label' or 'onehot'.")

    return df


# ─── Normalization ───────────────────────────────────────────────────────────

def normalize_features(
    df: pd.DataFrame,
    columns: list[str],
    method: NormalizationMethod = "standard",
) -> pd.DataFrame:
    """Scale numeric columns in-place using standard, minmax, or robust scaling."""
    df = df.copy()

    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric to normalize.")

        s = df[col]
        if method == "standard":
            std = s.std()
            df[col] = (s - s.mean()) / std if std != 0 else s - s.mean()
        elif method == "minmax":
            rng = s.max() - s.min()
            df[col] = (s - s.min()) / rng if rng != 0 else s * 0
        elif method == "robust":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            df[col] = (s - s.median()) / iqr if iqr != 0 else s - s.median()
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'standard', 'minmax', or 'robust'.")

    return df


# ─── Deduplication ───────────────────────────────────────────────────────────

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: Literal["first", "last", False] = "first",
) -> pd.DataFrame:
    """Remove duplicate rows, optionally scoped to a column subset."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = before - len(df)
    return df, removed
