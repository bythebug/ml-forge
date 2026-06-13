import numpy as np
import pandas as pd
import pytest

from features.feature_selector import (
    backward_elimination,
    forward_selection,
    information_value,
    statistical_selection,
    variance_threshold,
)
from features.normalizer import Scaler, log_transform, normalize, robust_scaling, standardize


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    age = rng.uniform(20, 60, n)
    income = age * 1500 + rng.normal(0, 5000, n)   # correlated with age
    noise = rng.normal(0, 1, n)                     # pure noise
    constant = np.ones(n)                            # zero variance
    target = age * 0.8 + income * 0.001 + rng.normal(0, 5, n)
    return pd.DataFrame({
        "age": age,
        "income": income,
        "noise": noise,
        "constant": constant,
        "target": target,
    })


@pytest.fixture
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    strong = rng.normal(0, 1, n)
    weak = rng.normal(0, 1, n)
    target = (strong + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"strong": strong, "weak": weak, "target": target})


@pytest.fixture
def train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    test = pd.DataFrame({"x": [10.0, 20.0], "y": [100.0, 200.0]})
    return train, test


# ─── variance threshold ───────────────────────────────────────────────────────

class TestVarianceThreshold:
    def test_removes_constant_column(self, regression_df):
        result = variance_threshold(regression_df, threshold=0.01)
        assert "constant" not in result["selected"]
        assert "constant" in result["dropped"]

    def test_keeps_high_variance_columns(self, regression_df):
        result = variance_threshold(regression_df, threshold=0.01)
        assert "age" in result["selected"]
        assert "income" in result["selected"]

    def test_returns_variance_scores(self, regression_df):
        result = variance_threshold(regression_df, threshold=0.01)
        assert "variances" in result
        assert "constant" in result["variances"]
        assert result["variances"]["constant"] == pytest.approx(0.0, abs=1e-10)

    def test_threshold_zero_keeps_everything(self, regression_df):
        result = variance_threshold(regression_df, threshold=0.0)
        assert set(result["dropped"]) == set()

    def test_high_threshold_drops_all(self, regression_df):
        result = variance_threshold(regression_df, threshold=1e9)
        assert len(result["selected"]) == 0


# ─── correlation / statistical selection ─────────────────────────────────────

class TestStatisticalSelection:
    def test_drops_noise_column(self, regression_df):
        result = statistical_selection(regression_df, target="target", threshold=0.3)
        assert "noise" not in result["selected"] or result["scores"]["noise"] >= 0.3

    def test_keeps_correlated_columns(self, regression_df):
        result = statistical_selection(regression_df, target="target", threshold=0.1)
        assert "age" in result["selected"]
        assert "income" in result["selected"]

    def test_scores_sorted_descending(self, regression_df):
        result = statistical_selection(regression_df, target="target", threshold=0.0)
        scores = list(result["scores"].values())
        assert scores == sorted(scores, reverse=True)

    def test_target_not_in_selected(self, regression_df):
        result = statistical_selection(regression_df, target="target", threshold=0.0)
        assert "target" not in result["selected"]

    def test_spearman_method(self, regression_df):
        result = statistical_selection(regression_df, target="target", method="spearman", threshold=0.0)
        assert result["method"] == "spearman"
        assert "age" in result["scores"]

    def test_threshold_zero_selects_all_numeric(self, regression_df):
        result = statistical_selection(regression_df, target="target", threshold=0.0)
        assert len(result["selected"]) == len(result["scores"])


# ─── forward selection ────────────────────────────────────────────────────────

class TestForwardSelection:
    def test_returns_at_most_max_features(self, regression_df):
        result = forward_selection(regression_df, target="target", max_features=2)
        assert len(result["selected"]) <= 2

    def test_first_selected_is_most_correlated(self, regression_df):
        result = forward_selection(regression_df, target="target", max_features=1)
        corr_result = statistical_selection(regression_df, target="target", threshold=0.0)
        top_feature = list(corr_result["scores"].keys())[0]
        assert result["selected"][0] == top_feature

    def test_r2_is_non_negative(self, regression_df):
        result = forward_selection(regression_df, target="target", max_features=3)
        assert result["final_r2"] >= 0

    def test_history_tracks_additions(self, regression_df):
        result = forward_selection(regression_df, target="target", max_features=2)
        assert len(result["selection_history"]) == len(result["selected"])
        for step in result["selection_history"]:
            assert "added" in step
            assert "r2" in step

    def test_target_not_in_selected(self, regression_df):
        result = forward_selection(regression_df, target="target", max_features=5)
        assert "target" not in result["selected"]


# ─── scaling methods ──────────────────────────────────────────────────────────

class TestScalingMethods:
    def test_standardize_mean_zero(self, train_test):
        train, _ = train_test
        result, _ = standardize(train, ["x"])
        assert abs(result["x"].mean()) < 1e-10

    def test_standardize_std_one(self, train_test):
        train, _ = train_test
        result, _ = standardize(train, ["x"])
        assert abs(result["x"].std(ddof=1) - 1.0) < 1e-10

    def test_normalize_range_zero_one(self, train_test):
        train, _ = train_test
        result, _ = normalize(train, ["x"])
        assert result["x"].min() == pytest.approx(0.0)
        assert result["x"].max() == pytest.approx(1.0)

    def test_robust_scaling_median_zero(self, train_test):
        train, _ = train_test
        result, _ = robust_scaling(train, ["x"])
        assert result["x"].median() == pytest.approx(0.0)

    def test_log_transform_non_negative(self, train_test):
        train, _ = train_test
        result, _ = log_transform(train, ["x"])
        assert (result["x"] >= 0).all()

    def test_inverse_transform_recovers_original(self, train_test):
        train, _ = train_test
        scaler = Scaler("standard").fit(train, ["x"])
        scaled = scaler.transform(train)
        recovered = scaler.inverse_transform(scaled)
        pd.testing.assert_series_equal(
            recovered["x"].round(8), train["x"].round(8), check_names=True
        )

    def test_inverse_minmax_recovers_original(self, train_test):
        train, _ = train_test
        scaler = Scaler("minmax").fit(train, ["x"])
        scaled = scaler.transform(train)
        recovered = scaler.inverse_transform(scaled)
        pd.testing.assert_series_equal(recovered["x"].round(8), train["x"].round(8))

    def test_transform_before_fit_raises(self):
        scaler = Scaler("standard")
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(pd.DataFrame({"x": [1.0]}))


# ─── train/test fit separation (data leakage prevention) ─────────────────────

class TestTrainTestFitSeparation:
    """
    The most critical test in this module. Verifies that:
      1. Scaler parameters are derived from training data only.
      2. The same parameters are applied to test data — not re-fit.
      3. Fitting on test data would yield different parameters (proof of separation).
    """

    def test_params_are_derived_from_train_only(self, train_test):
        train, test = train_test
        scaler = Scaler("standard").fit(train, ["x"])
        assert scaler.params["x"]["mean"] == pytest.approx(train["x"].mean())
        assert scaler.params["x"]["std"] == pytest.approx(train["x"].std(ddof=1))

    def test_test_uses_train_params_not_its_own(self, train_test):
        train, test = train_test
        scaler = Scaler("standard").fit(train, ["x"])
        test_transformed = scaler.transform(test)

        expected_val = (test["x"].iloc[0] - train["x"].mean()) / train["x"].std(ddof=1)
        assert test_transformed["x"].iloc[0] == pytest.approx(expected_val)

    def test_refitting_on_test_gives_different_params(self, train_test):
        train, test = train_test
        train_scaler = Scaler("standard").fit(train, ["x"])
        test_scaler = Scaler("standard").fit(test, ["x"])
        assert train_scaler.params["x"]["mean"] != pytest.approx(test_scaler.params["x"]["mean"])

    def test_standardize_convenience_with_train_df(self, train_test):
        train, test = train_test
        _, scaler = standardize(train, ["x"])
        test_scaled, _ = standardize(test, ["x"], train_df=train)

        expected = (test["x"].iloc[0] - scaler.params["x"]["mean"]) / scaler.params["x"]["std"]
        assert test_scaled["x"].iloc[0] == pytest.approx(expected)

    def test_scaler_serialization_round_trip(self, train_test):
        train, test = train_test
        original = Scaler("minmax").fit(train, ["x"])
        restored = Scaler.from_dict(original.to_dict())
        pd.testing.assert_frame_equal(
            original.transform(test).round(8),
            restored.transform(test).round(8),
        )

    def test_unknown_column_on_transform_raises(self, train_test):
        train, _ = train_test
        scaler = Scaler("standard").fit(train, ["x"])
        with pytest.raises(KeyError, match="not seen during fit"):
            scaler.transform(train, columns=["y"])

    def test_does_not_mutate_input(self, train_test):
        train, _ = train_test
        original_vals = train["x"].copy()
        scaler = Scaler("standard").fit(train, ["x"])
        scaler.transform(train)
        pd.testing.assert_series_equal(train["x"], original_vals)


# ─── information value ────────────────────────────────────────────────────────

class TestInformationValue:
    def test_strong_feature_has_higher_iv(self, binary_df):
        result = information_value(binary_df, target_col="target")
        assert "strong" in result
        assert "weak" in result
        assert result["strong"]["iv"] >= result["weak"]["iv"]

    def test_iv_structure(self, binary_df):
        result = information_value(binary_df, target_col="target")
        for col, data in result.items():
            assert "iv" in data
            assert "strength" in data
            assert "woe" in data

    def test_target_not_in_result(self, binary_df):
        result = information_value(binary_df, target_col="target")
        assert "target" not in result

    def test_sorted_by_iv_descending(self, binary_df):
        result = information_value(binary_df, target_col="target")
        ivs = [v["iv"] for v in result.values()]
        assert ivs == sorted(ivs, reverse=True)
