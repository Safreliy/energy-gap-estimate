"""Finite checks for the fixed-dictionary self-sparsification lemma.

These tests do not prove the population theorem.  They exhaust all sampling
outcomes in a small discrete model and check the algebraic invariants and the
two expected-risk estimates used by the proof.
"""

from itertools import product
import unittest

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


class SelfSparsificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [1.0, -0.5],
                [-0.25, 1.5],
                [0.75, 0.25],
                [-1.0, -0.75],
            ]
        )
        self.w = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.6, -0.8],
            ]
        )
        self.theta = np.array([0.7, -0.2, 0.5])
        self.q = 2
        self.features = relu(self.x @ self.w.T)
        self.predictor = self.features @ self.theta

    def outcomes(self):
        mass = np.linalg.norm(self.theta, ord=1)
        probabilities = np.abs(self.theta) / mass
        for indices in product(range(len(self.theta)), repeat=self.q):
            counts = np.bincount(indices, minlength=len(self.theta))
            sampled = mass * np.sign(self.theta) * counts / self.q
            probability = float(np.prod(probabilities[list(indices)]))
            yield probability, sampled, self.features @ sampled

    def test_exact_mass_support_and_unbiasedness(self) -> None:
        total_probability = 0.0
        mean_predictor = np.zeros_like(self.predictor)
        mass = np.linalg.norm(self.theta, ord=1)

        for probability, sampled, sampled_predictor in self.outcomes():
            total_probability += probability
            mean_predictor += probability * sampled_predictor
            self.assertLessEqual(np.count_nonzero(sampled), self.q)
            self.assertAlmostEqual(np.linalg.norm(sampled, ord=1), mass)

        self.assertAlmostEqual(total_probability, 1.0)
        np.testing.assert_allclose(mean_predictor, self.predictor, atol=1e-14)

    def test_mean_square_and_lipschitz_bounds(self) -> None:
        expected_mse = 0.0
        expected_l1_error = 0.0
        for probability, _, sampled_predictor in self.outcomes():
            error = sampled_predictor - self.predictor
            expected_mse += probability * float(np.mean(error**2))
            expected_l1_error += probability * float(np.mean(np.abs(error)))

        covariance = self.x.T @ self.x / len(self.x)
        d_x_squared = float(np.linalg.eigvalsh(covariance)[-1])
        mass = np.linalg.norm(self.theta, ord=1)
        variance_bound = mass**2 * d_x_squared / self.q

        self.assertLessEqual(expected_mse, variance_bound + 1e-14)
        self.assertLessEqual(expected_l1_error, np.sqrt(variance_bound) + 1e-14)

    def test_smooth_loss_linear_term_cancels(self) -> None:
        target = np.array([0.2, -0.4, 0.7, -0.1])
        base_gradient = self.predictor - target
        expected_linear_term = 0.0
        expected_objective_increment = 0.0
        expected_quadratic_remainder = 0.0

        for probability, _, sampled_predictor in self.outcomes():
            error = sampled_predictor - self.predictor
            expected_linear_term += probability * float(np.mean(base_gradient * error))
            expected_objective_increment += probability * float(
                np.mean(0.5 * (sampled_predictor - target) ** 2)
                - np.mean(0.5 * (self.predictor - target) ** 2)
            )
            expected_quadratic_remainder += probability * float(np.mean(0.5 * error**2))

        self.assertAlmostEqual(expected_linear_term, 0.0, places=14)
        self.assertAlmostEqual(
            expected_objective_increment,
            expected_quadratic_remainder,
            places=14,
        )

    def test_holder_loss_remainder_bound(self) -> None:
        alpha = 0.5
        power = 1.0 + alpha
        holder_constant = 2.0 ** (1.0 - alpha)
        target = np.array([0.2, -0.4, 0.7, -0.1])
        base_loss = np.mean(np.abs(self.predictor - target) ** power / power)
        expected_increment = 0.0

        for probability, _, sampled_predictor in self.outcomes():
            sampled_loss = np.mean(
                np.abs(sampled_predictor - target) ** power / power
            )
            expected_increment += probability * float(sampled_loss - base_loss)

        covariance = self.x.T @ self.x / len(self.x)
        d_x_squared = float(np.linalg.eigvalsh(covariance)[-1])
        mass = np.linalg.norm(self.theta, ord=1)
        variance_bound = mass**2 * d_x_squared / self.q
        holder_bound = (
            holder_constant
            / power
            * variance_bound ** (power / 2.0)
        )

        self.assertGreaterEqual(expected_increment, -1e-14)
        self.assertLessEqual(expected_increment, holder_bound + 1e-14)

    def test_smallest_mass_pruning_bound(self) -> None:
        l = 1
        q = len(self.theta) - l
        keep = np.argsort(np.abs(self.theta))[l:]
        pruned = np.zeros_like(self.theta)
        pruned[keep] = self.theta[keep]
        removed_mass = np.linalg.norm(self.theta - pruned, ord=1)

        self.assertLessEqual(np.count_nonzero(pruned), q)
        self.assertLessEqual(
            removed_mass,
            l * np.linalg.norm(self.theta, ord=1) / len(self.theta) + 1e-14,
        )

    def test_orthogonal_dictionary_sharpness_example(self) -> None:
        width = 8
        q = 4
        features = np.sqrt(width) * np.eye(width)
        theta = np.full(width, 1.0 / width)
        predictor = features @ theta
        sparse = np.zeros(width)
        sparse[:q] = theta[:q]
        error = features @ sparse - predictor

        self.assertAlmostEqual(
            float(np.mean(error**2)),
            (width - q) / width**2,
        )
        self.assertAlmostEqual(
            float(np.mean(np.abs(error))),
            (width - q) / (width * np.sqrt(width)),
        )


if __name__ == "__main__":
    unittest.main()
