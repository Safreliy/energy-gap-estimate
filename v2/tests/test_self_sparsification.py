from __future__ import annotations

import unittest

import numpy as np

from v2.experiments.core import EmpiricalObjective, ModelState
from v2.experiments.self_sparsification import (
    balanced_weight_decay_lift,
    raw_weight_decay_penalty,
    sample_on_fixed_dictionary,
    smallest_mass_pruning,
)


class SelfSparsificationTest(unittest.TestCase):
    def test_mse_objective_uses_half_squared_loss(self) -> None:
        X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]])
        y = np.array([0.2, -0.4, 0.1])
        state = ModelState(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0.3, -0.2]),
        )
        objective = EmpiricalObjective(X, y, loss="mse", kappa=0.07)
        prediction = np.maximum(X @ state.W.T, 0.0) @ state.theta
        expected = 0.5 * np.mean((prediction - y) ** 2) + 0.07 * np.sum(
            np.abs(state.theta)
        )
        self.assertAlmostEqual(objective.value(state), expected, places=14)
        self.assertEqual(objective.loss_smoothness, 1.0)
        with self.assertRaises(ValueError):
            _ = objective.loss_lipschitz

    def test_sampling_preserves_mass_and_support(self) -> None:
        theta = np.array([0.7, -0.2, 0.0, 0.5])
        batch = sample_on_fixed_dictionary(
            theta, q=2, trials=200, rng=np.random.default_rng(7)
        )
        np.testing.assert_allclose(
            np.sum(np.abs(batch.coefficients), axis=1),
            np.linalg.norm(theta, ord=1),
            atol=1e-14,
        )
        self.assertLessEqual(
            int(np.max(np.count_nonzero(batch.coefficients, axis=1))), 2
        )
        np.testing.assert_allclose(
            np.mean(batch.coefficients, axis=0), theta, atol=0.13
        )

    def test_smallest_mass_pruning(self) -> None:
        state = ModelState(np.eye(4), np.array([0.4, -0.1, 0.3, 0.2]))
        pruned = smallest_mass_pruning(state, 2)
        np.testing.assert_array_equal(pruned.theta, np.array([0.4, 0.0, 0.3, 0.0]))

    def test_balanced_lift_preserves_function_and_penalty(self) -> None:
        rng = np.random.default_rng(11)
        W = rng.normal(size=(5, 3))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        theta = np.array([0.7, -0.2, 0.0, 0.05, -0.9])
        atomic = ModelState(W, theta)
        raw = balanced_weight_decay_lift(atomic)
        X = rng.normal(size=(20, 3))
        atomic_prediction = np.maximum(X @ W.T, 0.0) @ theta
        raw_prediction = np.maximum(X @ raw.W.T, 0.0) @ raw.theta
        np.testing.assert_allclose(raw_prediction, atomic_prediction, atol=1e-14)
        self.assertAlmostEqual(
            raw_weight_decay_penalty(raw, 0.3),
            0.3 * np.linalg.norm(theta, ord=1),
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
