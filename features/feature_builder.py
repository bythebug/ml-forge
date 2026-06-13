"""
build_feature_set: dispatches a features_list JSON spec into engineered columns.

Spec format (matches FeatureSet.features_list JSONB schema):
{
  "features": [
    {"name": "age",                    "type": "numeric",     "source": "raw"},
    {"name": "age_squared",            "type": "numeric",     "source": "polynomial",  "base": "age",    "degree": 2},
    {"name": "age_x_income",           "type": "numeric",     "source": "interaction", "operands": ["age", "income"]},
    {"name": "age_lag_1",              "type": "numeric",     "source": "lag",         "base": "age",    "lag": 1},
    {"name": "income_rolling_7_mean",  "type": "numeric",     "source": "rolling",     "base": "income", "window": 7,  "agg": "mean"},
    {"name": "gender_encoded",         "type": "numeric",     "source": "encoding",    "base": "gender", "method": "label"},
    {"name": "signup_day_of_week",     "type": "categorical", "source": "time",        "base": "signup", "component": "day_of_week"},
    {"name": "age_bin",                "type": "categorical", "source": "binning",     "base": "age",    "bins": 5, "strategy": "quantile"},
    {"name": "income_by_city_mean",    "type": "numeric",     "source": "statistical", "base": "income", "group": "city", "agg": "mean"},
    {"name": "income_log",             "type": "numeric",     "source": "log",         "base": "income"},
    {"name": "revenue_per_visits",     "type": "numeric",     "source": "ratio",       "numerator": "revenue", "denominator": "visits"}
  ]
}
"""

from typing import Any

import numpy as np
import pandas as pd


def build_feature_set(df: pd.DataFrame, feature_definitions: dict) -> pd.DataFrame:
    """
    Apply all feature definitions in `feature_definitions["features"]` to `df`.
    Returns a new DataFrame with the engineered columns appended.
    Raw source features are validated to exist; all others are computed and named
    exactly as specified in the `name` field.
    """
    result = df.copy()
    for feat in feature_definitions.get("features", []):
        source = feat.get("source")
        name = feat["name"]

        try:
            series = _dispatch(result, feat, source)
        except KeyError as e:
            raise KeyError(
                f"Feature '{name}' (source='{source}') references missing column {e}."
            ) from e

        if series is not None:
            result[name] = series

    return result


def validate_feature_definitions(feature_definitions: dict) -> list[str]:
    """
    Return a list of validation error messages. Empty list = valid spec.
    Does not require a DataFrame — checks spec structure only.
    """
    errors: list[str] = []
    features = feature_definitions.get("features")
    if not isinstance(features, list):
        return ["'features' key must be a list."]

    required_source_fields: dict[str, list[str]] = {
        "raw":         ["name", "source"],
        "polynomial":  ["name", "source", "base"],
        "interaction": ["name", "source", "operands"],
        "lag":         ["name", "source", "base"],
        "rolling":     ["name", "source", "base"],
        "encoding":    ["name", "source", "base"],
        "time":        ["name", "source", "base", "component"],
        "binning":     ["name", "source", "base"],
        "statistical": ["name", "source", "base", "group"],
        "log":         ["name", "source", "base"],
        "ratio":       ["name", "source", "numerator", "denominator"],
    }

    seen_names: set[str] = set()
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            errors.append(f"Feature at index {i} must be a dict.")
            continue

        name = feat.get("name", f"<index {i}>")
        source = feat.get("source")

        if name in seen_names:
            errors.append(f"Duplicate feature name: '{name}'.")
        seen_names.add(name)

        if source not in required_source_fields:
            errors.append(f"Feature '{name}': unknown source '{source}'.")
            continue

        for field in required_source_fields[source]:
            if field not in feat:
                errors.append(f"Feature '{name}' (source='{source}'): missing field '{field}'.")

    return errors


# ─── dispatch ────────────────────────────────────────────────────────────────

def _dispatch(df: pd.DataFrame, feat: dict[str, Any], source: str) -> pd.Series | None:
    if source == "raw":
        if feat["name"] not in df.columns:
            raise KeyError(f"'{feat['name']}'")
        return None  # already in df

    if source == "polynomial":
        return df[feat["base"]] ** feat.get("degree", 2)

    if source == "interaction":
        ops = feat["operands"]
        result = df[ops[0]].copy()
        for op in ops[1:]:
            result = result * df[op]
        return result

    if source == "lag":
        return df[feat["base"]].shift(feat.get("lag", 1))

    if source == "rolling":
        rolled = df[feat["base"]].rolling(window=feat.get("window", 7), min_periods=1)
        return getattr(rolled, feat.get("agg", "mean"))()

    if source == "encoding":
        col = df[feat["base"]]
        method = feat.get("method", "label")
        if method == "label":
            return col.astype("category").cat.codes.where(col.notna(), other=-1)
        raise ValueError(f"Encoding method '{method}' not supported in build_feature_set. Use 'label'.")

    if source == "time":
        dt = pd.to_datetime(df[feat["base"]], errors="coerce")
        component = feat["component"]
        if component == "day_of_month":
            return dt.dt.day
        if component == "week_of_year":
            return dt.dt.isocalendar().week.astype("int64")
        if component == "is_weekend":
            return dt.dt.day_of_week >= 5
        return getattr(dt.dt, component)

    if source == "binning":
        strategy = feat.get("strategy", "uniform")
        bins = feat.get("bins", 10)
        if strategy == "quantile":
            return pd.qcut(df[feat["base"]], q=bins, labels=False, duplicates="drop")
        return pd.cut(df[feat["base"]], bins=bins, labels=False)

    if source == "statistical":
        agg = feat.get("agg", "mean")
        return df.groupby(feat["group"])[feat["base"]].transform(agg)

    if source == "log":
        return np.log1p(df[feat["base"]].clip(lower=0))

    if source == "ratio":
        return df[feat["numerator"]] / df[feat["denominator"]].replace(0, np.nan)

    raise ValueError(f"Unknown feature source: '{source}'.")
