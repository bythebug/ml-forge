"""
Stateful feature scaling — the Scaler class separates fit (train) from transform
(train + test) to prevent data leakage.

    CORRECT
    -------
    scaler = Scaler("standard").fit(X_train, columns)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)   # same params, not re-fit

    WRONG (leakage)
    ---------------
    scaler.fit(X_full, columns)   # test statistics contaminate train
"""

from typing import Literal, Optional

import numpy as np
import pandas as pd

ScalingMethod = Literal["standard", "minmax", "robust", "log"]


class Scaler:
    """
    Fit-once, transform-many scaler. Stores per-column statistics computed from
    the training set and reuses them for any subsequent transform call.
    """

    def __init__(self, method: ScalingMethod = "standard") -> None:
        if method not in ("standard", "minmax", "robust", "log"):
            raise ValueError(f"Unknown method '{method}'. Use: standard, minmax, robust, log.")
        self.method = method
        self.columns_: list[str] = []
        self._params: dict[str, dict] = {}
        self._fitted: bool = False

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "Scaler":
        """Compute and store scaling parameters from `df` (must be training data only)."""
        self.columns_ = list(columns)
        self._params = {}

        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be numeric.")

            s = df[col].dropna()

            if self.method == "standard":
                self._params[col] = {"mean": float(s.mean()), "std": float(s.std(ddof=1))}
            elif self.method == "minmax":
                self._params[col] = {"min": float(s.min()), "max": float(s.max())}
            elif self.method == "robust":
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                self._params[col] = {"median": float(s.median()), "iqr": q3 - q1}
            elif self.method == "log":
                self._params[col] = {}  # log1p is stateless

        self._fitted = True
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Apply fitted parameters to `df`. Safe to call on train and test."""
        self._assert_fitted()
        cols = columns or self.columns_
        self._assert_columns_known(cols)
        df = df.copy()

        for col in cols:
            p = self._params[col]
            if self.method == "standard":
                std = p["std"] or 1.0  # guard against zero std (constant column)
                df[col] = (df[col] - p["mean"]) / std
            elif self.method == "minmax":
                rng = (p["max"] - p["min"]) or 1.0
                df[col] = (df[col] - p["min"]) / rng
            elif self.method == "robust":
                iqr = p["iqr"] or 1.0
                df[col] = (df[col] - p["median"]) / iqr
            elif self.method == "log":
                df[col] = np.log1p(df[col].clip(lower=0))

        return df

    # ── fit_transform ─────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Convenience: fit on `df` then transform it. Use only for training data."""
        return self.fit(df, columns).transform(df)

    # ── inverse_transform ────────────────────────────────────────────────────

    def inverse_transform(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Reverse scaling to recover original-scale values."""
        self._assert_fitted()
        cols = columns or self.columns_
        self._assert_columns_known(cols)
        df = df.copy()

        for col in cols:
            p = self._params[col]
            if self.method == "standard":
                df[col] = df[col] * (p["std"] or 1.0) + p["mean"]
            elif self.method == "minmax":
                rng = (p["max"] - p["min"]) or 1.0
                df[col] = df[col] * rng + p["min"]
            elif self.method == "robust":
                df[col] = df[col] * (p["iqr"] or 1.0) + p["median"]
            elif self.method == "log":
                df[col] = np.expm1(df[col])

        return df

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize fitted parameters for storage (e.g., in FeatureSet JSONB)."""
        self._assert_fitted()
        return {"method": self.method, "columns": self.columns_, "params": self._params}

    @classmethod
    def from_dict(cls, data: dict) -> "Scaler":
        """Restore a Scaler from a serialized dict."""
        scaler = cls(method=data["method"])
        scaler.columns_ = data["columns"]
        scaler._params = data["params"]
        scaler._fitted = True
        return scaler

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def params(self) -> dict:
        return self._params

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── private guards ────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Scaler is not fitted. Call fit() with training data first.")

    def _assert_columns_known(self, cols: list[str]) -> None:
        unknown = [c for c in cols if c not in self._params]
        if unknown:
            raise KeyError(f"Columns not seen during fit: {unknown}")


# ── module-level convenience functions ────────────────────────────────────────
# Each returns (transformed_df, fitted_scaler) so the caller keeps the scaler
# for later use on test data.

def standardize(
    df: pd.DataFrame,
    columns: list[str],
    train_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Scaler]:
    """Z-score: mean=0, std=1. Pass train_df to fit on training data only."""
    scaler = Scaler("standard").fit(train_df if train_df is not None else df, columns)
    return scaler.transform(df), scaler


def normalize(
    df: pd.DataFrame,
    columns: list[str],
    train_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Scaler]:
    """Min-max: scale to [0, 1]. Pass train_df to fit on training data only."""
    scaler = Scaler("minmax").fit(train_df if train_df is not None else df, columns)
    return scaler.transform(df), scaler


def robust_scaling(
    df: pd.DataFrame,
    columns: list[str],
    train_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Scaler]:
    """Robust: (x - median) / IQR. Resistant to outliers."""
    scaler = Scaler("robust").fit(train_df if train_df is not None else df, columns)
    return scaler.transform(df), scaler


def log_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, Scaler]:
    """log1p transform — stateless, no train/test distinction needed."""
    scaler = Scaler("log").fit(df, columns)
    return scaler.transform(df), scaler
