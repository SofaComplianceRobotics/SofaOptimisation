"""Tests for score normalization and aggregation."""

import pytest

from sofaopt.core.scoring import aggregate_trial_scores, normalize_test_score


class TestNormalize:
    def test_divides_by_max(self):
        assert normalize_test_score(5.0, 10.0) == 0.5

    def test_clamps_above_max(self):
        assert normalize_test_score(15.0, 10.0) == 1.0

    def test_zero_max_gives_zero(self):
        assert normalize_test_score(5.0, 0.0) == 0.0


class TestRepeatAggregation:
    def test_empty_scores_give_zeros(self):
        assert aggregate_trial_scores([]) == (0.0, 0.0, 0.0, 0.0)

    def test_mean_is_default(self):
        agg, _, final, median = aggregate_trial_scores([1.0, 2.0, 6.0])
        assert agg == final == pytest.approx(3.0)
        assert median == 2.0

    def test_median_mode(self):
        agg, _, _, _ = aggregate_trial_scores([1.0, 2.0, 6.0], aggregation="median")
        assert agg == 2.0

    def test_sum_mode(self):
        agg, _, _, _ = aggregate_trial_scores([1.0, 2.0, 6.0], aggregation="sum")
        assert agg == 9.0

    def test_exponential_coverage_multiplies_per_positive_run(self):
        # 3 positive runs -> sum * 1.5^2
        agg, _, _, _ = aggregate_trial_scores(
            [1.0, 1.0, 2.0], aggregation="exponential_coverage"
        )
        assert agg == pytest.approx(4.0 * 2.25)

    def test_exponential_coverage_zero_runs_score_nothing(self):
        agg, _, _, _ = aggregate_trial_scores(
            [0.0, 0.0], aggregation="exponential_coverage"
        )
        assert agg == 0.0

    def test_exponential_coverage_ignores_zero_scores_in_count(self):
        # One positive run: multiplier stays 1.0 even with zero-score runs present.
        agg, _, _, _ = aggregate_trial_scores(
            [3.0, 0.0], aggregation="exponential_coverage"
        )
        assert agg == pytest.approx(3.0)


class TestWeightedCombination:
    def test_weighted_normalized_score_out_of_100(self):
        # a: 5/10 -> 0.5 * 40 = 20 ; b: 20/20 -> 1.0 * 60 = 60
        agg, _, final, _ = aggregate_trial_scores(
            [5.0, 20.0],
            weights={"a": 0.4, "b": 0.6},
            names=["a", "b"],
            max_scores={"a": 10.0, "b": 20.0},
        )
        assert final == pytest.approx(80.0)

    def test_scores_above_max_are_clamped(self):
        _, _, final, _ = aggregate_trial_scores(
            [50.0],
            weights={"a": 1.0},
            names=["a"],
            max_scores={"a": 10.0},
        )
        assert final == pytest.approx(100.0)
