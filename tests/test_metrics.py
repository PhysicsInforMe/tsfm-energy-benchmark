"""Tests for evaluation metrics."""

import numpy as np
import pytest

from energy_benchmark.evaluation.metrics import (
    mae,
    rmse,
    mase,
    weighted_quantile_loss,
)


class TestPointMetrics:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == 1.0

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-9


class TestMASE:
    def test_mase_equals_naive(self):
        """If the model is as good as seasonal naive, MASE should be ~1."""
        np.random.seed(42)
        n = 200
        # Add noise so seasonal differences are non-zero
        train = np.sin(np.arange(n) * 2 * np.pi / 24) * 100 + 1000
        train += np.random.randn(n) * 10
        # Naive forecast: shift by 24
        y_true = train[-48:-24]
        y_pred_naive = train[-72:-48]
        result = mase(y_true, y_pred_naive, train, seasonality=24)
        assert 0.5 < result < 3.0

    def test_mase_perfect_is_zero(self):
        np.random.seed(42)
        # Train data with noise so scaling factor > 0
        train = np.random.randn(100) * 10 + 100
        y_true = np.array([1.0, 2.0, 1.0, 2.0])
        assert mase(y_true, y_true, train, seasonality=2) == 0.0


class TestWQL:
    def test_wql_perfect_median(self):
        y_true = np.array([1.0, 2.0, 3.0])
        # All quantiles predict perfectly
        q_pred = np.column_stack([y_true, y_true, y_true])
        result = weighted_quantile_loss(y_true, q_pred, [0.1, 0.5, 0.9])
        assert abs(result) < 1e-9

    def test_wql_positive_for_errors(self):
        y_true = np.array([1.0, 2.0, 3.0])
        q_pred = np.column_stack([y_true + 5, y_true + 5, y_true + 5])
        result = weighted_quantile_loss(y_true, q_pred, [0.1, 0.5, 0.9])
        assert result > 0
