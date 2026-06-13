import numpy as np
import pandas as pd
import pytest

from features.feature_builder import build_feature_set, validate_feature_definitions
from features.features import (
    binning_features,
    categorical_encoding,
    domain_specific_features,
    interaction_features,
    lag_features,
    log_transform,
    numerical_features,
    polynomial_features,
    ratio_features,
    rolling_features,
    statistical_features,
    time_features,
)


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age":    [25.0, 30.0, 35.0, 40.0, 45.0],
            "income": [50_000.0, 60_000.0, 70_000.0, 80_000.0, 90_000.0],
            "city":   ["NY", "LA", "NY", "LA", "NY"],
            "gender": ["M", "F", "M", "F", "M"],
            "signup": pd.date_range("2023-01-02", periods=5),  # Monday–Friday
        }
    )


@pytest.fixture
def business_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "revenue":                  [1000.0, 2000.0, 1500.0],
            "num_orders":               [10.0, 20.0, 15.0],
            "visits":                   [100.0, 200.0, 150.0],
            "days_since_last_purchase": [5.0, 10.0, 1.0],
        }
    )


# ─── polynomial features ─────────────────────────────────────────────────────

class TestPolynomialFeatures:
    def test_adds_squared_column(self, base_df):
        result = polynomial_features(base_df, columns=["age"], degree=2)
        assert "age_pow2" in result.columns
        assert result["age_pow2"].iloc[0] == pytest.approx(625.0)

    def test_adds_cubic_column(self, base_df):
        result = polynomial_features(base_df, columns=["age"], degree=3)
        assert "age_pow2" in result.columns
        assert "age_pow3" in result.columns
        assert result["age_pow3"].iloc[0] == pytest.approx(15625.0)

    def test_multiple_columns(self, base_df):
        result = polynomial_features(base_df, columns=["age", "income"], degree=2)
        assert "age_pow2" in result.columns
        assert "income_pow2" in result.columns

    def test_raises_on_degree_less_than_2(self, base_df):
        with pytest.raises(ValueError, match="degree"):
            polynomial_features(base_df, columns=["age"], degree=1)

    def test_does_not_mutate_input(self, base_df):
        cols_before = base_df.columns.tolist()
        polynomial_features(base_df, columns=["age"])
        assert base_df.columns.tolist() == cols_before


# ─── interaction features ─────────────────────────────────────────────────────

class TestInteractionFeatures:
    def test_multiplies_two_columns(self, base_df):
        result = interaction_features(base_df, columns=["age", "income"])
        assert "age_x_income" in result.columns
        assert result["age_x_income"].iloc[0] == pytest.approx(25.0 * 50_000.0)

    def test_all_pairwise_combinations(self, base_df):
        result = interaction_features(base_df, columns=["age", "income"])
        assert len([c for c in result.columns if "_x_" in c]) == 1  # C(2,2)=1

    def test_explicit_pairs(self, base_df):
        result = interaction_features(base_df, pairs=[("age", "income")])
        assert "age_x_income" in result.columns

    def test_does_not_mutate_input(self, base_df):
        cols_before = set(base_df.columns)
        interaction_features(base_df, columns=["age", "income"])
        assert set(base_df.columns) == cols_before


# ─── lag features ─────────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_single_lag_shifts_correctly(self, base_df):
        result = lag_features(base_df, column="age", lags=[1])
        assert "age_lag_1" in result.columns
        assert pd.isna(result["age_lag_1"].iloc[0])
        assert result["age_lag_1"].iloc[1] == pytest.approx(base_df["age"].iloc[0])

    def test_multiple_lags_created(self, base_df):
        result = lag_features(base_df, column="age", lags=[1, 2, 3])
        for lag in [1, 2, 3]:
            assert f"age_lag_{lag}" in result.columns

    def test_lag_7_creates_all_nans_on_small_df(self, base_df):
        result = lag_features(base_df, column="age", lags=[7])
        assert result["age_lag_7"].isna().all()

    def test_does_not_mutate_input(self, base_df):
        cols_before = set(base_df.columns)
        lag_features(base_df, column="age")
        assert set(base_df.columns) == cols_before


# ─── rolling features ─────────────────────────────────────────────────────────

class TestRollingFeatures:
    def test_adds_rolling_mean(self, base_df):
        result = rolling_features(base_df, column="age", windows=[3])
        assert "age_rolling_3_mean" in result.columns

    def test_adds_rolling_std_and_sum(self, base_df):
        result = rolling_features(base_df, column="age", windows=[3], agg_fns=["std", "sum"])
        assert "age_rolling_3_std" in result.columns
        assert "age_rolling_3_sum" in result.columns

    def test_rolling_mean_value_correct(self, base_df):
        result = rolling_features(base_df, column="age", windows=[3], agg_fns=["mean"])
        # idx 2: mean(25,30,35) = 30
        assert result["age_rolling_3_mean"].iloc[2] == pytest.approx(30.0)

    def test_multiple_windows(self, base_df):
        result = rolling_features(base_df, column="age", windows=[2, 3], agg_fns=["mean"])
        assert "age_rolling_2_mean" in result.columns
        assert "age_rolling_3_mean" in result.columns


# ─── categorical encoding ──────────────────────────────────────────────────────

class TestCategoricalEncoding:
    def test_label_encoding_adds_encoded_column(self, base_df):
        result = categorical_encoding(base_df, columns=["city"], method="label")
        assert "city_encoded" in result.columns
        assert pd.api.types.is_integer_dtype(result["city_encoded"])

    def test_label_encoding_preserves_original_column(self, base_df):
        result = categorical_encoding(base_df, columns=["city"], method="label")
        assert "city" in result.columns

    def test_onehot_expands_to_binary_columns(self, base_df):
        result = categorical_encoding(base_df, columns=["city"], method="onehot")
        assert "city" not in result.columns
        city_cols = [c for c in result.columns if c.startswith("city_")]
        assert len(city_cols) == base_df["city"].nunique()

    def test_onehot_values_are_binary(self, base_df):
        result = categorical_encoding(base_df, columns=["city"], method="onehot")
        city_cols = [c for c in result.columns if c.startswith("city_")]
        for col in city_cols:
            assert set(result[col].unique()).issubset({0, 1})

    def test_raises_on_unknown_method(self, base_df):
        with pytest.raises(ValueError, match="Unknown encoding method"):
            categorical_encoding(base_df, columns=["city"], method="target")


# ─── time features ────────────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_adds_day_of_week(self, base_df):
        result = time_features(base_df, date_column="signup", components=["day_of_week"])
        assert "signup_day_of_week" in result.columns

    def test_monday_is_zero(self, base_df):
        # 2023-01-02 is a Monday
        result = time_features(base_df, date_column="signup", components=["day_of_week"])
        assert result["signup_day_of_week"].iloc[0] == 0

    def test_adds_is_weekend(self, base_df):
        result = time_features(base_df, date_column="signup")
        assert "signup_is_weekend" in result.columns

    def test_weekdays_not_weekend(self, base_df):
        # All 5 dates are Mon–Fri
        result = time_features(base_df, date_column="signup", components=["day_of_week"])
        assert not result["signup_is_weekend"].any()

    def test_adds_month_and_year(self, base_df):
        result = time_features(base_df, date_column="signup", components=["month", "year"])
        assert "signup_month" in result.columns
        assert "signup_year" in result.columns
        assert (result["signup_year"] == 2023).all()


# ─── binning features ─────────────────────────────────────────────────────────

class TestBinningFeatures:
    def test_uniform_binning_creates_column(self, base_df):
        result = binning_features(base_df, column="age", bins=5, strategy="uniform")
        assert "age_bin_uniform_5" in result.columns

    def test_uniform_bins_within_range(self, base_df):
        result = binning_features(base_df, column="age", bins=3, strategy="uniform")
        valid = result["age_bin_uniform_3"].dropna()
        assert valid.min() >= 0
        assert valid.max() < 3

    def test_quantile_binning_creates_column(self, base_df):
        result = binning_features(base_df, column="age", bins=5, strategy="quantile")
        assert "age_bin_quantile_5" in result.columns

    def test_raises_on_unknown_strategy(self, base_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            binning_features(base_df, column="age", strategy="kmeans")


# ─── statistical features ─────────────────────────────────────────────────────

class TestStatisticalFeatures:
    def test_adds_group_mean(self, base_df):
        result = statistical_features(base_df, target_col="income", group_cols=["city"], agg_fns=["mean"])
        assert "income_by_city_mean" in result.columns

    def test_group_mean_correct(self, base_df):
        result = statistical_features(base_df, target_col="income", group_cols=["city"], agg_fns=["mean"])
        ny_mean = base_df[base_df["city"] == "NY"]["income"].mean()
        ny_rows = result[result["city"] == "NY"]["income_by_city_mean"]
        assert (ny_rows == pytest.approx(ny_mean)).all()

    def test_multiple_group_cols(self, base_df):
        result = statistical_features(base_df, target_col="income", group_cols=["city", "gender"], agg_fns=["mean"])
        assert "income_by_city_mean" in result.columns
        assert "income_by_gender_mean" in result.columns

    def test_count_agg(self, base_df):
        result = statistical_features(base_df, target_col="income", group_cols=["city"], agg_fns=["count"])
        assert "income_by_city_count" in result.columns


# ─── feature set building ─────────────────────────────────────────────────────

class TestFeatureSetBuilding:
    @pytest.fixture
    def spec(self):
        return {
            "features": [
                {"name": "age",           "type": "numeric",     "source": "raw"},
                {"name": "age_sq",        "type": "numeric",     "source": "polynomial",  "base": "age",    "degree": 2},
                {"name": "age_x_income",  "type": "numeric",     "source": "interaction", "operands": ["age", "income"]},
                {"name": "income_log",    "type": "numeric",     "source": "log",         "base": "income"},
                {"name": "income_lag_1",  "type": "numeric",     "source": "lag",         "base": "income", "lag": 1},
                {"name": "city_encoded",  "type": "numeric",     "source": "encoding",    "base": "city",   "method": "label"},
                {"name": "signup_month",  "type": "categorical", "source": "time",        "base": "signup", "component": "month"},
                {"name": "income_by_city","type": "numeric",     "source": "statistical", "base": "income", "group": "city", "agg": "mean"},
                {"name": "age_bin",       "type": "categorical", "source": "binning",     "base": "age",    "bins": 3, "strategy": "uniform"},
            ]
        }

    def test_all_columns_present(self, base_df, spec):
        result = build_feature_set(base_df, spec)
        expected = ["age_sq", "age_x_income", "income_log", "income_lag_1",
                    "city_encoded", "signup_month", "income_by_city", "age_bin"]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"

    def test_raw_column_not_duplicated(self, base_df, spec):
        result = build_feature_set(base_df, spec)
        assert result.columns.tolist().count("age") == 1

    def test_polynomial_value_correct(self, base_df, spec):
        result = build_feature_set(base_df, spec)
        assert result["age_sq"].iloc[0] == pytest.approx(625.0)

    def test_does_not_mutate_input(self, base_df, spec):
        cols_before = base_df.columns.tolist()
        build_feature_set(base_df, spec)
        assert base_df.columns.tolist() == cols_before

    def test_missing_raw_column_raises(self, base_df):
        spec = {"features": [{"name": "missing_col", "type": "numeric", "source": "raw"}]}
        with pytest.raises(KeyError):
            build_feature_set(base_df, spec)

    def test_unknown_source_raises(self, base_df):
        spec = {"features": [{"name": "x", "type": "numeric", "source": "fourier", "base": "age"}]}
        with pytest.raises(ValueError, match="Unknown feature source"):
            build_feature_set(base_df, spec)


class TestValidateFeatureDefinitions:
    def test_valid_spec_returns_no_errors(self):
        spec = {"features": [{"name": "x", "source": "raw"}]}
        assert validate_feature_definitions(spec) == []

    def test_duplicate_name_flagged(self):
        spec = {"features": [
            {"name": "x", "source": "raw"},
            {"name": "x", "source": "raw"},
        ]}
        errors = validate_feature_definitions(spec)
        assert any("Duplicate" in e for e in errors)

    def test_missing_required_field_flagged(self):
        spec = {"features": [{"name": "x", "source": "polynomial"}]}  # missing "base"
        errors = validate_feature_definitions(spec)
        assert any("base" in e for e in errors)

    def test_unknown_source_flagged(self):
        spec = {"features": [{"name": "x", "source": "fourier"}]}
        errors = validate_feature_definitions(spec)
        assert any("unknown source" in e for e in errors)
