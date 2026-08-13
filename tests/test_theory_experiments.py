from __future__ import annotations

import unittest

import numpy as np

from theory_experiments.core import (
    EmpiricalObjective,
    ModelState,
    compress_by_cluster_merge,
    compress_by_nearest_cluster_merge,
    initialize_state,
    interpolate,
    make_ridge_teacher_dataset,
    sphericalize_state,
    TrainConfig,
    train_projected_adam,
)
from theory_experiments.paths import construct_proof_path, evaluate_proof_path, evaluate_segment


class TheoryExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.uniform(-1.0, 1.0, size=(40, 2))
        y = rng.normal(scale=0.2, size=40)
        self.objective = EmpiricalObjective(
            X, y, loss="huber", huber_delta=0.25, kappa=0.01
        )

    def test_projection_domain_and_interpolation(self) -> None:
        rng = np.random.default_rng(5)
        a = initialize_state(7, 2, rng)
        b = initialize_state(7, 2, rng)
        for t in np.linspace(0.0, 1.0, 31):
            state = interpolate(a, b, float(t))
            self.assertLessEqual(np.max(np.linalg.norm(state.W, axis=1)), 1.0 + 1e-12)

    def test_cluster_merge_support_l1_and_bound(self) -> None:
        rng = np.random.default_rng(6)
        state = initialize_state(12, 2, rng)
        state.theta += np.sign(state.theta) * 0.1
        result = compress_by_cluster_merge(state, reserve_neurons=4)
        self.assertLessEqual(result.support_after, 8)
        self.assertLessEqual(
            np.sum(np.abs(result.state.theta)), np.sum(np.abs(state.theta)) + 1e-12
        )
        before = self.objective.value(state)
        after = self.objective.value(result.state)
        bound = (
            4.0
            * np.sqrt(2.0)
            * self.objective.loss_lipschitz
            * self.objective.d_x
            * np.sum(np.abs(state.theta))
            * ((4 + 1) / 12) ** 0.5
        )
        self.assertLessEqual(after - before, bound + 1e-10)

    def test_sphericalization_preserves_predictor_and_reduces_l1(self) -> None:
        state = ModelState(
            W=np.asarray([[0.3, 0.4], [0.0, 0.0], [-0.2, 0.0]]),
            theta=np.asarray([2.0, 3.0, -1.5]),
        )
        prediction = self.objective.predict(state)
        transformed = sphericalize_state(state)
        np.testing.assert_allclose(
            self.objective.predict(transformed), prediction, atol=1e-12
        )
        self.assertLessEqual(
            np.sum(np.abs(transformed.theta)), np.sum(np.abs(state.theta))
        )
        active = np.abs(transformed.theta) > 1e-12
        np.testing.assert_allclose(
            np.linalg.norm(transformed.W[active], axis=1), 1.0, atol=1e-12
        )

    def test_nearest_cluster_merge_forces_nontrivial_compression(self) -> None:
        angles = np.linspace(-0.02, 0.02, 8)
        W = np.stack(
            [np.asarray([np.cos(angle), np.sin(angle)]) for angle in angles]
        )
        state = ModelState(W=W, theta=np.linspace(0.1, 0.8, 8))
        result = compress_by_nearest_cluster_merge(state, reserve_neurons=3)
        self.assertEqual(result.support_before, 8)
        self.assertLessEqual(result.support_after, 5)
        before = self.objective.value(state)
        after = self.objective.value(result.state)
        bound = (
            self.objective.loss_lipschitz
            * self.objective.d_x
            * np.sum(np.abs(state.theta))
            * result.cluster_diameter
        )
        self.assertLessEqual(after - before, bound + 1e-12)

    def test_segment_certificate_dominates_dense_scan(self) -> None:
        rng = np.random.default_rng(7)
        a = initialize_state(6, 2, rng)
        b = initialize_state(6, 2, rng)
        certificate = evaluate_segment(self.objective, a, b, grid_points=17)
        dense_max = max(
            self.objective.value(interpolate(a, b, float(t)))
            for t in np.linspace(0.0, 1.0, 10001)
        )
        self.assertLessEqual(dense_max, certificate.certified_upper + 1e-12)

    def test_constant_predictor_canonicalization_segments(self) -> None:
        rng = np.random.default_rng(8)
        left = initialize_state(10, 2, rng)
        right = initialize_state(10, 2, rng)
        reference = initialize_state(3, 2, rng)
        construction = construct_proof_path(left, right, reference)
        for state in construction.nodes:
            self.objective.validate_state(state)
        self.assertTrue(np.array_equal(construction.nodes[0].W, left.W))
        self.assertTrue(np.array_equal(construction.nodes[-1].W, right.W))
        self.assertTrue(np.array_equal(construction.nodes[0].theta, left.theta))
        self.assertTrue(np.array_equal(construction.nodes[-1].theta, right.theta))
        path_upper = evaluate_proof_path(self.objective, construction)
        dense_max = max(
            self.objective.value(interpolate(a, b, float(t)))
            for a, b in zip(construction.nodes[:-1], construction.nodes[1:])
            for t in np.linspace(0.0, 1.0, 101)
        )
        self.assertLessEqual(dense_max, path_upper + 1e-10)

    def test_optimizer_returns_feasible_improvement(self) -> None:
        rng = np.random.default_rng(9)
        initial = initialize_state(8, 2, rng)
        initial_value = self.objective.value(initial)
        result = train_projected_adam(
            self.objective,
            initial,
            TrainConfig(epochs=120, learning_rate=0.015, patience=40),
        )
        self.assertLessEqual(result.objective, initial_value)
        self.objective.validate_state(result.state)

    def test_logistic_objective_is_finite_and_globally_lipschitz(self) -> None:
        X = self.objective.X
        y = (self.objective.y > 0.0).astype(float)
        objective = EmpiricalObjective(X, y, loss="logistic", kappa=0.01)
        state = ModelState(
            W=np.zeros((3, X.shape[1])),
            theta=np.array([1e3, -1e3, 2e3], dtype=float),
        )
        self.assertTrue(np.isfinite(objective.value(state)))
        self.assertEqual(objective.loss_lipschitz, 1.0)

    def test_binary_teacher_is_reproducible_and_logistic_compatible(self) -> None:
        kwargs = dict(
            input_dim=2,
            n_samples=80,
            teacher_width=32,
            coefficient_decay=1.35,
            seed=20260812,
            target_scale=0.8,
            target_mode="binary",
        )
        X_a, y_a, metadata_a = make_ridge_teacher_dataset(**kwargs)
        X_b, y_b, metadata_b = make_ridge_teacher_dataset(**kwargs)
        self.assertTrue(np.array_equal(X_a, X_b))
        self.assertTrue(np.array_equal(y_a, y_b))
        self.assertEqual(set(np.unique(y_a)).difference({0.0, 1.0}), set())
        self.assertEqual(metadata_a, metadata_b)
        objective = EmpiricalObjective(X_a, y_a, loss="logistic", kappa=0.002)
        self.assertAlmostEqual(objective.loss_lipschitz, 1.0)

    def test_torch_backend_matches_numpy_cpu_in_float64(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            self.skipTest("PyTorch is not installed in this environment.")
        from theory_experiments.torch_backend import train_projected_adam_batch_torch

        rng = np.random.default_rng(10)
        initials = [initialize_state(5, 2, rng) for _ in range(2)]
        config = TrainConfig(
            epochs=35,
            learning_rate=0.01,
            patience=0,
            min_delta=0.0,
        )
        numpy_results = [
            train_projected_adam(self.objective, state, config) for state in initials
        ]
        torch_results = train_projected_adam_batch_torch(
            self.objective, initials, config, device="cpu", dtype="float64"
        )
        for numpy_result, torch_result in zip(numpy_results, torch_results):
            self.assertTrue(
                np.allclose(numpy_result.state.W, torch_result.state.W, atol=1e-9)
            )
            self.assertTrue(
                np.allclose(
                    numpy_result.state.theta, torch_result.state.theta, atol=1e-9
                )
            )
            self.assertAlmostEqual(
                numpy_result.objective, torch_result.objective, places=9
            )

    def test_torch_backend_early_stopping_branch(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            self.skipTest("PyTorch is not installed in this environment.")
        from theory_experiments.torch_backend import train_projected_adam_batch_torch

        rng = np.random.default_rng(11)
        initials = [initialize_state(4, 2, rng) for _ in range(2)]
        results = train_projected_adam_batch_torch(
            self.objective,
            initials,
            TrainConfig(
                epochs=10,
                learning_rate=0.0,
                patience=1,
                min_delta=1e-12,
            ),
            device="cpu",
            dtype="float64",
        )
        self.assertTrue(all(result.stop_reason == "patience" for result in results))

    def test_torch_float32_output_is_feasible_in_numpy(self) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            self.skipTest("PyTorch is not installed in this environment.")
        from theory_experiments.torch_backend import train_projected_adam_batch_torch

        rng = np.random.default_rng(12)
        initial = initialize_state(8, 2, rng)
        result = train_projected_adam_batch_torch(
            self.objective,
            [initial],
            TrainConfig(epochs=20, learning_rate=0.02, patience=0),
            device="cpu",
            dtype="float32",
        )[0]
        self.objective.validate_state(result.state)
        self.assertLessEqual(np.max(np.linalg.norm(result.state.W, axis=1)), 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
