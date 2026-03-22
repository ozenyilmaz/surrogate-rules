"""
tests/unit/test_rules_and_kpi.py
---------------------------------
Unit tests for rule evaluation and KPI computation.
"""

import numpy as np
import pytest

from src.rules.forest import (
    RulePath,
    check_path_representativeness,
    check_tree_representativeness,
    compute_rule_stats,
    compute_surrogate_score,
    sample_satisfies_path,
    sorensen_dice,
)
from src.kpi.metrics import (
    compute_expressive_power,
    get_coverage_and_predictions,
    hamming_loss_rule_model,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def simple_paths() -> list:
    """Two non-overlapping rules on feature 0."""
    return [
        [(0, 0.5, "L")],   # rule 0: x0 <= 0.5
        [(0, 0.5, "R")],   # rule 1: x0 > 0.5
    ]


@pytest.fixture
def simple_X() -> np.ndarray:
    return np.array([[0.2], [0.8], [0.3], [0.9]])


@pytest.fixture
def simple_y() -> np.ndarray:
    return np.array([-1, 1, -1, 1])


# ─────────────────────────────────────────────
# sample_satisfies_path
# ─────────────────────────────────────────────

class TestSampleSatisfiesPath:

    def test_left_condition_true(self):
        path: RulePath = [(0, 0.5, "L")]
        x = np.array([0.3])
        assert sample_satisfies_path(x, path) is True

    def test_left_condition_false(self):
        path: RulePath = [(0, 0.5, "L")]
        x = np.array([0.7])
        assert sample_satisfies_path(x, path) is False

    def test_right_condition_true(self):
        path: RulePath = [(0, 0.5, "R")]
        x = np.array([0.7])
        assert sample_satisfies_path(x, path) is True

    def test_empty_path_always_true(self):
        assert sample_satisfies_path(np.array([99.0]), []) is True

    def test_multi_condition_all_true(self):
        path: RulePath = [(0, 1.0, "L"), (1, 2.0, "R")]
        x = np.array([0.5, 3.0])
        assert sample_satisfies_path(x, path) is True

    def test_multi_condition_one_false(self):
        path: RulePath = [(0, 1.0, "L"), (1, 2.0, "R")]
        x = np.array([0.5, 1.0])  # 1.0 is NOT > 2.0
        assert sample_satisfies_path(x, path) is False


# ─────────────────────────────────────────────
# compute_rule_stats
# ─────────────────────────────────────────────

class TestComputeRuleStats:

    def test_basic_stats_shape(self, simple_X, simple_y, simple_paths):
        stats = compute_rule_stats(simple_X, simple_y, simple_paths)
        assert stats is not None
        assert len(stats.paths) == 2
        assert len(stats.labels) == 2
        assert len(stats.loss) == 2
        assert len(stats.weights) == 2

    def test_labels_are_in_valid_set(self, simple_X, simple_y, simple_paths):
        stats = compute_rule_stats(simple_X, simple_y, simple_paths)
        for label in stats.labels:
            assert label in {-1, 0, 1}

    def test_pruning_removes_small_rules(self, simple_X, simple_y, simple_paths):
        # min_samples_fraction=0.9 → need 90% of 4 = 4 samples; neither rule covers that many
        stats = compute_rule_stats(simple_X, simple_y, simple_paths, min_samples_fraction=0.9)
        assert stats is None


# ─────────────────────────────────────────────
# compute_surrogate_score
# ─────────────────────────────────────────────

class TestComputeSurrogateScore:

    def test_perfect_rules(self, simple_X, simple_y, simple_paths):
        labels = np.array([-1.0, 1.0])
        weights = np.array([2.0, 2.0])
        acc = compute_surrogate_score(simple_X, simple_y, simple_paths, labels, weights)
        assert acc == 1.0

    def test_wrong_labels_zero_acc(self, simple_X, simple_y, simple_paths):
        labels = np.array([1.0, -1.0])   # inverted
        weights = np.array([2.0, 2.0])
        acc = compute_surrogate_score(simple_X, simple_y, simple_paths, labels, weights)
        assert acc == 0.0


# ─────────────────────────────────────────────
# get_coverage_and_predictions
# ─────────────────────────────────────────────

class TestCoverageAndPredictions:

    def test_all_covered(self, simple_X, simple_paths):
        labels = np.array([-1.0, 1.0])
        coverage, preds = get_coverage_and_predictions(simple_X, simple_paths, labels)
        assert len(coverage) == len(simple_X)
        for cov in coverage:
            assert len(cov) > 0, "Every sample should be covered by exactly one rule"

    def test_no_coverage_returns_minus_one(self):
        X = np.array([[5.0]])   # no rule fires
        paths = [[(0, 0.5, "L")]]  # only covers x <= 0.5
        labels = np.array([-1.0])
        coverage, preds = get_coverage_and_predictions(X, paths, labels)
        assert preds[0] == -1


# ─────────────────────────────────────────────
# hamming_loss_rule_model
# ─────────────────────────────────────────────

class TestHammingLoss:

    def test_zero_loss_when_perfectly_covered(self):
        y = np.array([1, -1])
        coverage = [[0], []]  # positive covered, negative NOT covered by any rule
        loss = hamming_loss_rule_model(y, coverage)
        assert loss == 0.0

    def test_max_loss(self):
        y = np.array([1])
        coverage = [[]]  # positive sample not covered → penalty 1
        loss = hamming_loss_rule_model(y, coverage)
        assert loss == 1.0


# ─────────────────────────────────────────────
# sorensen_dice
# ─────────────────────────────────────────────

class TestSorensenDice:

    def test_identical_sets(self):
        assert sorensen_dice([1, 2, 3], [1, 2, 3]) == 1.0

    def test_disjoint_sets(self):
        assert sorensen_dice([1, 2], [3, 4]) == 0.0

    def test_empty_denominator(self):
        assert sorensen_dice([], []) == 0.0


# ─────────────────────────────────────────────
# compute_expressive_power
# ─────────────────────────────────────────────

class TestExpressivePower:

    def test_single_rule(self):
        paths = [[(0, 0.5, "L"), (1, 1.0, "R")]]
        metrics = compute_expressive_power(paths)
        assert metrics.rule_condition_size == 2.0
        assert metrics.feature_interaction_extent == 1.0

    def test_empty_paths(self):
        metrics = compute_expressive_power([])
        assert metrics.rule_condition_size == 0.0
        assert metrics.feature_interaction_extent == 0.0
