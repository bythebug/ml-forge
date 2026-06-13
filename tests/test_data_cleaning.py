import numpy as np
import pandas as pd
import pytest

from data.preprocessor import (
    detect_outliers,
    encode_categoricals,
    handle_missing_values,
    normalize_features,
    remove_duplicates,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age":    [25.0, 30.0, np.nan, 45.0, 200.0],  # 200 is outlier
            "income": [50_000.0, np.nan, 70_000.0, 80_000.0, 90_000.0],
            "gender": ["M", "F", "M", np.nan, "F"],
            "city":   ["NY", "LA", "NY", "LA", "NY"],
        }
    )


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})


# ─── missing value handling ───────────────────────────────────────────────────

class TestMissingValueHandling:
    def test_drop_strategy_removes_rows(self, base_df):
        result = handle_missing_values(base_df, strategy="drop")
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(base_df)

    def test_mean_strategy_fills_numeric(self, base_df):
        result = handle_missing_values(base_df, strategy="mean", columns=["age", "income"])
        assert result["age"].isnull().sum() == 0
        assert result["income"].isnull().sum() == 0

    def test_mean_strategy_ignores_non_numeric(self, base_df):
        result = handle_missing_values(base_df, strategy="mean", columns=["gender"])
        assert result["gender"].isnull().sum() == base_df["gender"].isnull().sum()

    def test_median_strategy(self, base_df):
        result = handle_missing_values(base_df, strategy="median", columns=["age"])
        assert result["age"].isnull().sum() == 0
        assert result["age"].iloc[2] == pd.Series([25.0, 30.0, 45.0, 200.0]).median()

    def test_mode_strategy_fills_categorical(self, base_df):
        result = handle_missing_values(base_df, strategy="mode", columns=["gender"])
        assert result["gender"].isnull().sum() == 0

    def test_forward_fill(self, base_df):
        result = handle_missing_values(base_df, strategy="forward_fill", columns=["income"])
        assert result["income"].iloc[1] == 50_000.0

    def test_constant_fill(self, base_df):
        result = handle_missing_values(base_df, strategy="constant", columns=["age"], fill_value=-1)
        assert result["age"].iloc[2] == -1.0

    def test_constant_requires_fill_value(self, base_df):
        with pytest.raises(ValueError, match="fill_value"):
            handle_missing_values(base_df, strategy="constant", columns=["age"])

    def test_does_not_mutate_input(self, base_df):
        original_nulls = base_df.isnull().sum().sum()
        handle_missing_values(base_df, strategy="mean")
        assert base_df.isnull().sum().sum() == original_nulls


# ─── outlier detection ────────────────────────────────────────────────────────

class TestOutlierDetection:
    def test_iqr_detects_extreme_value(self, base_df):
        result = detect_outliers(base_df, column="age", method="iqr")
        assert result["outlier_count"] >= 1
        assert 4 in result["outlier_indices"]  # index 4 has age=200

    def test_zscore_detects_extreme_value(self):
        # Need enough "normal" points so the extreme value is clearly > 3 std devs
        df = pd.DataFrame({"age": [1.0] * 30 + [200.0]})
        result = detect_outliers(df, column="age", method="zscore")
        assert result["outlier_count"] >= 1

    def test_result_contains_expected_keys(self, base_df):
        result = detect_outliers(base_df, column="age")
        assert {"column", "method", "lower_bound", "upper_bound", "outlier_count", "outlier_pct", "outlier_indices"} <= result.keys()

    def test_no_outliers_in_uniform_data(self, numeric_df):
        result = detect_outliers(numeric_df, column="x", method="iqr")
        assert result["outlier_count"] == 0

    def test_raises_on_non_numeric_column(self, base_df):
        with pytest.raises(TypeError, match="numeric"):
            detect_outliers(base_df, column="gender")

    def test_raises_on_unknown_method(self, base_df):
        with pytest.raises(ValueError, match="Unknown method"):
            detect_outliers(base_df, column="age", method="invalid")


# ─── normalization ────────────────────────────────────────────────────────────

class TestNormalization:
    def test_standard_scaling_mean_zero(self, numeric_df):
        result = normalize_features(numeric_df, columns=["x"], method="standard")
        assert abs(result["x"].mean()) < 1e-10

    def test_standard_scaling_std_one(self, numeric_df):
        result = normalize_features(numeric_df, columns=["x"], method="standard")
        assert abs(result["x"].std(ddof=1) - 1.0) < 1e-10

    def test_minmax_bounds_zero_to_one(self, numeric_df):
        result = normalize_features(numeric_df, columns=["x"], method="minmax")
        assert result["x"].min() == pytest.approx(0.0)
        assert result["x"].max() == pytest.approx(1.0)

    def test_robust_scaling(self, numeric_df):
        result = normalize_features(numeric_df, columns=["x"], method="robust")
        assert result["x"].median() == pytest.approx(0.0)

    def test_does_not_mutate_input(self, numeric_df):
        original = numeric_df["x"].copy()
        normalize_features(numeric_df, columns=["x"])
        pd.testing.assert_series_equal(numeric_df["x"], original)

    def test_raises_on_non_numeric(self, base_df):
        with pytest.raises(TypeError, match="numeric"):
            normalize_features(base_df, columns=["gender"])

    def test_raises_on_unknown_method(self, numeric_df):
        with pytest.raises(ValueError, match="Unknown method"):
            normalize_features(numeric_df, columns=["x"], method="log")


# ─── categorical encoding ─────────────────────────────────────────────────────

class TestCategoricalEncoding:
    def test_label_encoding_returns_integers(self, base_df):
        result = encode_categoricals(base_df, columns=["gender"], method="label")
        assert pd.api.types.is_integer_dtype(result["gender"])

    def test_label_encoding_preserves_row_count(self, base_df):
        result = encode_categoricals(base_df, columns=["city"], method="label")
        assert len(result) == len(base_df)

    def test_onehot_expands_columns(self, base_df):
        result = encode_categoricals(base_df, columns=["city"], method="onehot")
        assert "city_NY" in result.columns or any("city" in c for c in result.columns)
        assert "city" not in result.columns

    def test_onehot_no_information_loss(self, base_df):
        result = encode_categoricals(base_df, columns=["city"], method="onehot")
        city_cols = [c for c in result.columns if c.startswith("city_")]
        assert len(city_cols) == base_df["city"].nunique()

    def test_does_not_mutate_input(self, base_df):
        original_dtype = base_df["gender"].dtype
        encode_categoricals(base_df, columns=["gender"], method="label")
        assert base_df["gender"].dtype == original_dtype

    def test_raises_on_unknown_method(self, base_df):
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encode_categoricals(base_df, columns=["gender"], method="binary")


# ─── deduplication ───────────────────────────────────────────────────────────

class TestDeduplication:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result, removed = remove_duplicates(df)
        assert len(result) == 2
        assert removed == 1

    def test_subset_deduplication(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        result, removed = remove_duplicates(df, subset=["a"])
        assert len(result) == 2
        assert removed == 1

    def test_no_duplicates_returns_same(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, removed = remove_duplicates(df)
        assert removed == 0
        assert len(result) == 3
