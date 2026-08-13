from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class ModelState:
    """Parameters of the bias-free model sum_i theta_i ReLU(w_i^T x)."""

    W: Array
    theta: Array

    def copy(self) -> "ModelState":
        return ModelState(self.W.copy(), self.theta.copy())

    @property
    def width(self) -> int:
        return int(self.theta.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.W.shape[1])


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2500
    learning_rate: float = 0.02
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    patience: int = 250
    min_delta: float = 1e-9
    target_objective: float | None = None
    log_every: int = 0


@dataclass
class TrainResult:
    state: ModelState
    objective: float
    data_loss: float
    l1_penalty: float
    epochs_run: int
    stop_reason: str
    history: list[float]


@dataclass
class CompressionResult:
    state: ModelState
    support_before: int
    support_after: int
    merged_indices: list[int]
    representative: int | None
    cluster_diameter: float


class EmpiricalObjective:
    """Finite-distribution analogue of the objective in the manuscript.

    The empirical distribution puts mass 1/N on every row of ``X``.  Thus this
    is an expectation, not an approximation hidden inside the path evaluator.
    Supported losses are globally Lipschitz in the scalar prediction.
    """

    def __init__(
        self,
        X: Array,
        y: Array,
        *,
        loss: str,
        kappa: float,
        huber_delta: float = 0.25,
    ) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be two-dimensional and aligned with y.")
        if len(X) == 0:
            raise ValueError("The empirical distribution cannot be empty.")
        if loss not in {"huber", "logistic"}:
            raise ValueError("loss must be 'huber' or 'logistic'.")
        if kappa <= 0:
            raise ValueError("kappa must be positive.")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        if loss == "logistic" and np.any((y < 0) | (y > 1)):
            raise ValueError("Logistic targets must lie in [0, 1].")
        self.X = X
        self.y = y
        self.loss = loss
        self.kappa = float(kappa)
        self.huber_delta = float(huber_delta)

    @property
    def input_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def loss_lipschitz(self) -> float:
        return self.huber_delta if self.loss == "huber" else 1.0

    @property
    def d_x(self) -> float:
        sigma = self.X.T @ self.X / len(self.X)
        return float(np.sqrt(np.linalg.eigvalsh(sigma)[-1]))

    def validate_state(self, state: ModelState, tol: float = 1e-10) -> None:
        if state.W.ndim != 2 or state.theta.ndim != 1:
            raise ValueError("W must be a matrix and theta a vector.")
        if state.W.shape != (len(state.theta), self.input_dim):
            raise ValueError("State dimensions do not match the objective.")
        if np.max(np.linalg.norm(state.W, axis=1), initial=0.0) > 1.0 + tol:
            raise ValueError("A first-layer row lies outside the unit ball.")

    def predict(self, state: ModelState) -> Array:
        self.validate_state(state)
        return np.maximum(self.X @ state.W.T, 0.0) @ state.theta

    def _loss_values_and_derivative(self, prediction: Array) -> tuple[Array, Array]:
        if self.loss == "huber":
            residual = prediction - self.y
            absolute = np.abs(residual)
            quadratic = absolute <= self.huber_delta
            values = np.where(
                quadratic,
                0.5 * residual**2,
                self.huber_delta * (absolute - 0.5 * self.huber_delta),
            )
            derivative = np.clip(residual, -self.huber_delta, self.huber_delta)
            return values, derivative

        # log(1 + exp(z)) - y z, evaluated without overflow.
        values = np.logaddexp(0.0, prediction) - self.y * prediction
        positive = prediction >= 0
        sigmoid = np.empty_like(prediction)
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-prediction[positive]))
        exp_z = np.exp(prediction[~positive])
        sigmoid[~positive] = exp_z / (1.0 + exp_z)
        return values, sigmoid - self.y

    def components(self, state: ModelState) -> tuple[float, float, float]:
        prediction = self.predict(state)
        values, _ = self._loss_values_and_derivative(prediction)
        data_loss = float(np.mean(values))
        l1_penalty = float(self.kappa * np.sum(np.abs(state.theta)))
        return data_loss + l1_penalty, data_loss, l1_penalty

    def value(self, state: ModelState) -> float:
        return self.components(state)[0]

    def data_gradients(self, state: ModelState) -> tuple[Array, Array]:
        """Gradient of empirical loss; the L1 term is handled proximally."""
        self.validate_state(state)
        preactivation = self.X @ state.W.T
        activation = np.maximum(preactivation, 0.0)
        prediction = activation @ state.theta
        _, derivative = self._loss_values_and_derivative(prediction)
        derivative = derivative / len(self.X)
        grad_theta = activation.T @ derivative
        gate = preactivation > 0.0
        weighted = derivative[:, None] * state.theta[None, :] * gate
        grad_W = weighted.T @ self.X
        return grad_W, grad_theta

    def segment_lipschitz(self, a: ModelState, b: ModelState) -> float:
        """A certified Lipschitz constant for t -> F((1-t)a+t b).

        It uses the global Lipschitz constant of the loss and the fact that the
        ReLU of a convex combination is bounded by the largest endpoint ReLU.
        """
        self.validate_state(a)
        self.validate_state(b)
        if a.W.shape != b.W.shape:
            raise ValueError("Segment endpoints must have identical shapes.")
        delta_W = b.W - a.W
        delta_theta = b.theta - a.theta
        relu_a = np.maximum(self.X @ a.W.T, 0.0)
        relu_b = np.maximum(self.X @ b.W.T, 0.0)
        max_relu = np.maximum(relu_a, relu_b)
        theta_term = max_relu @ np.abs(delta_theta)
        movement = np.abs(self.X @ delta_W.T)
        weight_term = movement @ np.maximum(np.abs(a.theta), np.abs(b.theta))
        predictor_lipschitz = float(np.mean(theta_term + weight_term))
        penalty_lipschitz = self.kappa * float(np.sum(np.abs(delta_theta)))
        return self.loss_lipschitz * predictor_lipschitz + penalty_lipschitz


def project_first_layer(W: Array) -> Array:
    W = np.asarray(W, dtype=np.float64)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    scale = np.maximum(1.0, norms)
    return W / scale


def sphericalize_state(
    state: ModelState,
    *,
    active_tolerance: float = 1e-12,
) -> ModelState:
    """Move every active nonzero ridge atom to the unit sphere exactly.

    Positive homogeneity gives
    ``theta * ReLU(w.T @ x) = (theta * ||w||) * ReLU((w/||w||).T @ x)``.
    Hence the returned state has the same predictor and no larger output
    ``L1`` norm. An active coefficient at the zero row contributes nothing and
    is set to zero. The function returns the endpoint of the monotone path used
    in the sphere-cover proof; it does not discretize that path.
    """
    result = state.copy()
    active = np.flatnonzero(np.abs(result.theta) > active_tolerance)
    for index in active.tolist():
        radius = float(np.linalg.norm(result.W[index]))
        if radius <= active_tolerance:
            result.W[index] = 0.0
            result.theta[index] = 0.0
            continue
        result.W[index] /= radius
        result.theta[index] *= radius
    return result


def interpolate(a: ModelState, b: ModelState, t: float) -> ModelState:
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1].")
    return ModelState((1.0 - t) * a.W + t * b.W, (1.0 - t) * a.theta + t * b.theta)


def initialize_state(width: int, input_dim: int, rng: np.random.Generator) -> ModelState:
    W = rng.normal(size=(width, input_dim))
    W = project_first_layer(W)
    radii = rng.uniform(0.35, 1.0, size=(width, 1))
    W *= radii
    theta = rng.normal(scale=0.05 / np.sqrt(max(1, width)), size=width)
    return ModelState(W, theta)


def train_projected_adam(
    objective: EmpiricalObjective,
    initial: ModelState,
    config: TrainConfig,
) -> TrainResult:
    """Full-batch projected proximal Adam inside the theorem's parameter set."""
    state = initial.copy()
    objective.validate_state(state)
    m_W = np.zeros_like(state.W)
    v_W = np.zeros_like(state.W)
    m_theta = np.zeros_like(state.theta)
    v_theta = np.zeros_like(state.theta)
    best_state = state.copy()
    best_value, best_data, best_l1 = objective.components(state)
    history = [best_value]
    stale = 0
    stop_reason = "max_epochs"

    for epoch in range(1, config.epochs + 1):
        grad_W, grad_theta = objective.data_gradients(state)
        m_W = config.beta1 * m_W + (1.0 - config.beta1) * grad_W
        v_W = config.beta2 * v_W + (1.0 - config.beta2) * grad_W**2
        m_theta = config.beta1 * m_theta + (1.0 - config.beta1) * grad_theta
        v_theta = config.beta2 * v_theta + (1.0 - config.beta2) * grad_theta**2

        correction1 = 1.0 - config.beta1**epoch
        correction2 = 1.0 - config.beta2**epoch
        state.W -= config.learning_rate * (m_W / correction1) / (
            np.sqrt(v_W / correction2) + config.adam_eps
        )
        state.W = project_first_layer(state.W)
        theta_metric = np.sqrt(v_theta / correction2) + config.adam_eps
        state.theta -= config.learning_rate * (m_theta / correction1) / theta_metric
        # Diagonal-metric proximal step for kappa ||theta||_1.  Exact zeros
        # make the active-neuron count observable and avoid the ambiguous
        # subgradient-at-zero behavior of ordinary Adam.
        shrink = config.learning_rate * objective.kappa / theta_metric
        state.theta = np.sign(state.theta) * np.maximum(
            np.abs(state.theta) - shrink, 0.0
        )

        value, data_loss, l1_penalty = objective.components(state)
        history.append(value)
        if value < best_value - config.min_delta:
            best_value = value
            best_data = data_loss
            best_l1 = l1_penalty
            best_state = state.copy()
            stale = 0
        else:
            stale += 1

        if config.target_objective is not None and best_value <= config.target_objective:
            stop_reason = "target"
            break
        if config.patience > 0 and stale >= config.patience:
            stop_reason = "patience"
            break
        if config.log_every and epoch % config.log_every == 0:
            print(f"epoch={epoch} objective={value:.8f} best={best_value:.8f}")

    return TrainResult(
        state=best_state,
        objective=float(best_value),
        data_loss=float(best_data),
        l1_penalty=float(best_l1),
        epochs_run=epoch,
        stop_reason=stop_reason,
        history=history,
    )


def compress_by_cluster_merge(
    state: ModelState,
    *,
    reserve_neurons: int,
    active_tolerance: float = 1e-12,
) -> CompressionResult:
    """Implement the explicit ambient-cube merge certificate.

    The returned state has at most ``m - reserve_neurons`` active output
    coefficients.  Its first layer is unchanged and its L1 norm cannot grow.
    """
    m, n = state.W.shape
    if not 1 <= reserve_neurons < m:
        raise ValueError("reserve_neurons must satisfy 1 <= l < m.")
    active = np.flatnonzero(np.abs(state.theta) > active_tolerance)
    support_before = int(len(active))
    target_support = m - reserve_neurons
    if support_before <= target_support:
        return CompressionResult(
            state=state.copy(),
            support_before=support_before,
            support_after=support_before,
            merged_indices=[],
            representative=None,
            cluster_diameter=0.0,
        )

    k = support_before - target_support
    p = max(1, int(np.floor((support_before / (k + 1)) ** (1.0 / n))))
    # Deterministic cells in [-1, 1]^n, matching the proof rather than a
    # data-dependent k-means clustering.
    scaled = np.floor((state.W[active] + 1.0) * (p / 2.0)).astype(int)
    scaled = np.clip(scaled, 0, p - 1)
    cells: dict[tuple[int, ...], list[int]] = {}
    for index, cell in zip(active.tolist(), scaled.tolist()):
        cells.setdefault(tuple(cell), []).append(int(index))
    cluster = max(cells.values(), key=lambda values: (len(values), -min(values)))
    if len(cluster) < k + 1:
        raise RuntimeError("Pigeonhole cluster was smaller than the proof guarantees.")

    representative = int(cluster[0])
    compressed = state.copy()
    compressed.theta[representative] = float(np.sum(state.theta[cluster]))
    for index in cluster[1:]:
        compressed.theta[index] = 0.0
    support_after = int(np.count_nonzero(np.abs(compressed.theta) > active_tolerance))
    if support_after > target_support:
        raise RuntimeError("Cluster merge failed to reach the requested support.")
    points = state.W[cluster]
    diameter = 0.0
    if len(points) > 1:
        diameter = float(
            np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2))
        )
    return CompressionResult(
        state=compressed,
        support_before=support_before,
        support_after=support_after,
        merged_indices=[int(i) for i in cluster],
        representative=representative,
        cluster_diameter=diameter,
    )


def compress_by_nearest_cluster_merge(
    state: ModelState,
    *,
    reserve_neurons: int,
    active_tolerance: float = 1e-12,
) -> CompressionResult:
    """Select and merge a genuinely close cluster under a support budget.

    If ``k`` coefficients must be removed, every active atom is paired with
    its ``k`` nearest active neighbours and the smallest-diameter candidate is
    selected. This deterministic diagnostic is not an estimator of the
    worst-case covering radius; the observed cluster diameter gives its own
    post-hoc analytic merge certificate.
    """
    m = state.width
    if not 1 <= reserve_neurons < m:
        raise ValueError("reserve_neurons must satisfy 1 <= l < m.")
    active = np.flatnonzero(np.abs(state.theta) > active_tolerance)
    support_before = int(len(active))
    target_support = m - reserve_neurons
    if support_before <= target_support:
        return CompressionResult(
            state=state.copy(),
            support_before=support_before,
            support_after=support_before,
            merged_indices=[],
            representative=None,
            cluster_diameter=0.0,
        )

    k = support_before - target_support
    points = state.W[active]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for row in range(support_before):
        nearest_local = np.argsort(distances[row], kind="stable")[: k + 1]
        cluster_local = tuple(sorted(int(item) for item in nearest_local))
        cluster_points = points[list(cluster_local)]
        diameter = float(
            np.max(
                np.linalg.norm(
                    cluster_points[:, None, :] - cluster_points[None, :, :],
                    axis=2,
                ),
                initial=0.0,
            )
        )
        candidates.append((diameter, cluster_local))

    diameter, selected_local = min(candidates, key=lambda item: (item[0], item[1]))
    cluster = [int(active[item]) for item in selected_local]
    representative = min(cluster)
    compressed = state.copy()
    compressed.theta[representative] = float(np.sum(state.theta[cluster]))
    for index in cluster:
        if index != representative:
            compressed.theta[index] = 0.0
    support_after = int(np.count_nonzero(np.abs(compressed.theta) > active_tolerance))
    if support_after > target_support:
        raise RuntimeError("Nearest-cluster merge failed to reach the support budget.")
    return CompressionResult(
        state=compressed,
        support_before=support_before,
        support_after=support_after,
        merged_indices=cluster,
        representative=representative,
        cluster_diameter=float(diameter),
    )


def make_ridge_teacher_dataset(
    *,
    input_dim: int,
    n_samples: int,
    teacher_width: int,
    coefficient_decay: float,
    seed: int,
    target_scale: float = 1.0,
    target_mode: str = "regression",
) -> tuple[Array, Array, dict[str, Any]]:
    """Generate a bounded finite distribution from a wide ridge-function teacher.

    ``target_mode="regression"`` returns the normalized teacher logit.  The
    ``"binary"`` mode thresholds that same latent logit at zero, so the
    binary-cross-entropy run changes the loss/labels without changing the
    constrained architecture or the underlying ridge-function teacher.
    """
    if input_dim < 1 or n_samples < 2 or teacher_width < 1:
        raise ValueError("Invalid synthetic dataset dimensions.")
    if target_mode not in {"regression", "binary"}:
        raise ValueError("target_mode must be 'regression' or 'binary'.")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_samples, input_dim))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    # Uniform radii in the Euclidean unit ball.  In dimension one this retains
    # a nontrivial continuum of inputs instead of collapsing to {-1, +1}.
    radii = rng.uniform(size=(n_samples, 1)) ** (1.0 / input_dim)
    X = radii * directions
    teacher_W = rng.normal(size=(teacher_width, input_dim))
    teacher_W /= np.maximum(np.linalg.norm(teacher_W, axis=1, keepdims=True), 1e-12)
    ranks = np.arange(1, teacher_width + 1, dtype=np.float64)
    signs = rng.choice(np.array([-1.0, 1.0]), size=teacher_width)
    teacher_theta = signs * ranks ** (-coefficient_decay)
    rng.shuffle(teacher_theta)
    raw = np.maximum(X @ teacher_W.T, 0.0) @ teacher_theta
    raw -= float(np.mean(raw))
    standard_deviation = float(np.std(raw))
    if standard_deviation <= 1e-12:
        raise RuntimeError("Degenerate teacher realization.")
    latent = target_scale * raw / standard_deviation
    if target_mode == "regression":
        y = latent
    else:
        y = (latent >= 0.0).astype(np.float64)
    metadata = {
        "type": f"ridge_teacher_{target_mode}",
        "target_mode": target_mode,
        "input_dim": input_dim,
        "n_samples": n_samples,
        "teacher_width": teacher_width,
        "coefficient_decay": coefficient_decay,
        "seed": seed,
        "target_scale": target_scale,
        "max_input_norm": float(np.max(np.linalg.norm(X, axis=1))),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
    }
    if target_mode == "binary":
        metadata["positive_fraction"] = float(np.mean(y))
    return X, y, metadata
