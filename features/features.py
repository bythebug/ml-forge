"""
12 feature engineering techniques — each function adds new columns to a copy of the
input DataFrame and returns it. Input is never mutated.
"""

from itertools import combinations
from typing import Literal, Optional

import numpy as np
import pandas as pd

AggFn = Literal["mean", "std", "sum", "min", "max", "median", "count"]
EncodingMethod = Literal["label", "onehot"]
BinStrategy = Literal["uniform", "quantile"]


# ── 1. Numerical passthrough ──────────────────────────────────────────────────

def numerical_features(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return a DataFrame containing only the specified (or all) numeric columns."""
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric columns requested: {non_numeric}")
    return df[cols].copy()


# ── 2. Polynomial features ────────────────────────────────────────────────────

def polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
) -> pd.DataFrame:
    """Add x^2 … x^degree for each column. Degree must be ≥ 2."""
    if degree < 2:
        raise ValueError("degree must be ≥ 2.")
    df = df.copy()
    for col in columns:
        for d in range(2, degree + 1):
            df[f"{col}_pow{d}"] = df[col] ** d
    return df


# ── 3. Interaction features ───────────────────────────────────────────────────

def interaction_features(
    df: pd.DataFrame,
    columns: list[str],
    pairs: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Multiply column pairs. If `pairs` is given, compute only those; otherwise
    compute all pairwise combinations of `columns`.
    """
    df = df.copy()
    target_pairs = pairs or list(combinations(columns, 2))
    for col1, col2 in target_pairs:
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    return df


# ── 4. Lag features ───────────────────────────────────────────────────────────

def lag_features(
    df: pd.DataFrame,
    column: str,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Shift `column` by each value in `lags` to create look-back features.
    Requires the DataFrame to be sorted by time before calling.
    """
    lags = lags or [1, 7, 30]
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


# ── 5. Rolling window features ────────────────────────────────────────────────

def rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int] = None,
    agg_fns: list[AggFn] = None,
) -> pd.DataFrame:
    """Rolling mean, std, and sum for each window size."""
    windows = windows or [7, 30]
    agg_fns = agg_fns or ["mean", "std", "sum"]
    df = df.copy()
    for w in windows:
        rolled = df[column].rolling(window=w, min_periods=1)
        for fn in agg_fns:
            df[f"{column}_rolling_{w}_{fn}"] = getattr(rolled, fn)()
    return df


# ── 6. Categorical encoding ───────────────────────────────────────────────────

def categorical_encoding(
    df: pd.DataFrame,
    columns: list[str],
    method: EncodingMethod = "onehot",
) -> pd.DataFrame:
    """
    Label-encode (ordinal int codes) or one-hot encode categorical columns.
    For one-hot, original columns are dropped and replaced with binary dummies.
    """
    df = df.copy()
    if method == "label":
        for col in columns:
            df[f"{col}_encoded"] = df[col].astype("category").cat.codes.where(
                df[col].notna(), other=-1
            )
    elif method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=int)
    else:
        raise ValueError(f"Unknown encoding method '{method}'. Use 'label' or 'onehot'.")
    return df


# ── 7. Domain-specific features ───────────────────────────────────────────────

def domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business logic features. Each block is guarded by a column-existence check so
    the function is safe to call on any DataFrame — it only adds what it can compute.

    Implemented formulas:
      avg_order_value      = revenue / num_orders
      recency_score        = 1 / (1 + days_since_last_purchase)   ∈ (0, 1]
      revenue_per_visit    = revenue / visits
      purchase_frequency   = num_orders / customer_age_days
      customer_lifetime_value = avg_order_value * purchase_frequency * avg_lifespan_days
    """
    df = df.copy()

    if {"revenue", "num_orders"}.issubset(df.columns):
        df["avg_order_value"] = df["revenue"] / df["num_orders"].replace(0, np.nan)

    if "days_since_last_purchase" in df.columns:
        df["recency_score"] = 1 / (1 + df["days_since_last_purchase"])

    if {"revenue", "visits"}.issubset(df.columns):
        df["revenue_per_visit"] = df["revenue"] / df["visits"].replace(0, np.nan)

    if {"num_orders", "customer_age_days"}.issubset(df.columns):
        df["purchase_frequency"] = df["num_orders"] / df["customer_age_days"].replace(0, np.nan)

    if {"avg_order_value", "purchase_frequency", "avg_lifespan_days"}.issubset(df.columns):
        df["customer_lifetime_value"] = (
            df["avg_order_value"] * df["purchase_frequency"] * df["avg_lifespan_days"]
        )

    return df


# ── 8. Time / date features ───────────────────────────────────────────────────

_TIME_COMPONENTS = {
    "day_of_week",    # 0 = Monday
    "day_of_month",
    "month",
    "quarter",
    "year",
    "hour",
    "week_of_year",
    "is_month_start",
    "is_month_end",
}


def time_features(
    df: pd.DataFrame,
    date_column: str,
    components: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Extract date/time components from `date_column`.
    `components` defaults to all standard components; pass a subset to be selective.
    Also adds `is_weekend` (bool) derived from day_of_week.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_column], errors="coerce")

    targets = components or list(_TIME_COMPONENTS)
    for comp in targets:
        if comp == "day_of_month":
            df[f"{date_column}_{comp}"] = dt.dt.day
        elif comp == "week_of_year":
            df[f"{date_column}_{comp}"] = dt.dt.isocalendar().week.astype("int64")
        else:
            df[f"{date_column}_{comp}"] = getattr(dt.dt, comp)

    if "day_of_week" in targets or components is None:
        df[f"{date_column}_is_weekend"] = dt.dt.day_of_week >= 5

    return df


# ── 9. Binning / discretization ───────────────────────────────────────────────

def binning_features(
    df: pd.DataFrame,
    column: str,
    bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> pd.DataFrame:
    """
    Discretize a continuous column into `bins` intervals.
    - 'uniform'  → equal-width bins (pd.cut)
    - 'quantile' → equal-frequency bins (pd.qcut)
    """
    df = df.copy()
    col_name = f"{column}_bin_{strategy}_{bins}"
    if strategy == "uniform":
        df[col_name] = pd.cut(df[column], bins=bins, labels=False)
    elif strategy == "quantile":
        df[col_name] = pd.qcut(df[column], q=bins, labels=False, duplicates="drop")
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'.")
    return df


# ── 10. Statistical / group aggregation features ──────────────────────────────

def statistical_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    agg_fns: list[AggFn] = None,
) -> pd.DataFrame:
    """
    For each combination of group columns, compute aggregation statistics of
    `target_col` and broadcast them back to every row (transform).
    Produces columns like: {target_col}_by_{group_col}_{agg_fn}
    """
    agg_fns = agg_fns or ["mean", "std", "count"]
    df = df.copy()
    for group_col in group_cols:
        for fn in agg_fns:
            col_name = f"{target_col}_by_{group_col}_{fn}"
            df[col_name] = df.groupby(group_col)[target_col].transform(fn)
    return df


# ── 11. Ratio features ────────────────────────────────────────────────────────

def ratio_features(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Compute numerator / denominator for each pair. Division by zero → NaN.
    Column name: {numerator}_per_{denominator}
    """
    df = df.copy()
    for numerator, denominator in pairs:
        col_name = f"{numerator}_per_{denominator}"
        df[col_name] = df[numerator] / df[denominator].replace(0, np.nan)
    return df


# ── 12. Log transform ─────────────────────────────────────────────────────────

def log_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply log1p to reduce right skew. Clips negatives to 0 before transforming
    so the function is safe on columns with occasional negative noise.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    return df
