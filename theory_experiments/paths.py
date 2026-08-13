from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .core import (
    CompressionResult,
    EmpiricalObjective,
    ModelState,
    TrainConfig,
    compress_by_cluster_merge,
    interpolate,
    train_projected_adam,
)


@dataclass(frozen=True)
class DSSConfig:
    max_depth: int = 6
    initial_grid_points: int = 33
    max_grid_points: int = 257
    final_grid_points: int = 257
    certificate_tolerance: float = 1e-5
    midpoint_train: TrainConfig = TrainConfig(epochs=1200, patience=150)


@dataclass
class SegmentEvaluation:
    sampled_max: float
    certified_upper: float
    argmax_t: float
    peak: ModelState
    lipschitz_constant: float
    grid_points: int


@dataclass
class SegmentRecord:
    depth: int
    evaluation: SegmentEvaluation
    status: str


@dataclass
class PathResult:
    nodes: list[ModelState]
    sampled_max: float
    certified_upper: float
    certificate_width: float
    threshold: float
    sampled_gap: float
    certified_gap_upper: float
    certified_below_level: bool
    unresolved_segments: int
    trained_midpoints: int
    records: list[SegmentRecord]


@dataclass
class ProofPathConstruction:
    nodes: list[ModelState]
    left_compression: CompressionResult
    right_compression: CompressionResult
    bridge_width: int


def evaluate_proof_path(
    objective: EmpiricalObjective,
    construction: ProofPathConstruction,
) -> float:
    """Exact analytic upper bound for the constructed piecewise path.

    Every segment is one of: output interpolation at fixed first layer
    (objective convex), motion of a zero-output row (objective constant), or
    coefficient transfer between identical atoms (objective constant).  Hence
    no grid discretization is required and the largest node value bounds the
    entire continuous path.
    """
    return float(max(objective.value(node) for node in construction.nodes))


def evaluate_segment(
    objective: EmpiricalObjective,
    a: ModelState,
    b: ModelState,
    *,
    grid_points: int,
) -> SegmentEvaluation:
    if grid_points < 2:
        raise ValueError("grid_points must be at least two.")
    grid = np.linspace(0.0, 1.0, int(grid_points))
    values = np.asarray([objective.value(interpolate(a, b, float(t))) for t in grid])
    index = int(np.argmax(values))
    sampled_max = float(values[index])
    lipschitz = objective.segment_lipschitz(a, b)
    mesh = 1.0 / (len(grid) - 1)
    certified_upper = sampled_max + 0.5 * mesh * lipschitz
    return SegmentEvaluation(
        sampled_max=sampled_max,
        certified_upper=float(certified_upper),
        argmax_t=float(grid[index]),
        peak=interpolate(a, b, float(grid[index])),
        lipschitz_constant=float(lipschitz),
        grid_points=len(grid),
    )


def _adaptive_evaluate(
    objective: EmpiricalObjective,
    a: ModelState,
    b: ModelState,
    threshold: float,
    config: DSSConfig,
) -> SegmentEvaluation:
    points = max(3, int(config.initial_grid_points))
    if points % 2 == 0:
        points += 1
    while True:
        evaluation = evaluate_segment(objective, a, b, grid_points=points)
        decisive = (
            evaluation.sampled_max > threshold + config.certificate_tolerance
            or evaluation.certified_upper <= threshold + config.certificate_tolerance
        )
        if decisive or points >= config.max_grid_points:
            return evaluation
        points = min(config.max_grid_points, 2 * points - 1)


def evaluate_piecewise_path(
    objective: EmpiricalObjective,
    nodes: list[ModelState],
    *,
    grid_points: int,
) -> tuple[float, float]:
    if len(nodes) < 2:
        raise ValueError("A path needs at least two nodes.")
    evaluations = [
        evaluate_segment(objective, a, b, grid_points=grid_points)
        for a, b in zip(nodes[:-1], nodes[1:])
    ]
    return (
        max(item.sampled_max for item in evaluations),
        max(item.certified_upper for item in evaluations),
    )


def _append_if_changed(nodes: list[ModelState], candidate: ModelState) -> None:
    previous = nodes[-1]
    if not (
        np.array_equal(previous.W, candidate.W)
        and np.array_equal(previous.theta, candidate.theta)
    ):
        nodes.append(candidate)


def _pack_active_atoms(
    state: ModelState,
    *,
    active_tolerance: float,
) -> list[ModelState]:
    """Continuously pack active atoms into the first coordinates.

    Every segment produced here has exactly constant predictor and L1 norm.
    An inactive row is moved to an active atom, its coefficient is transferred
    between identical rows, and the newly inactive row is reset to zero.
    """
    current = state.copy()
    nodes = [current.copy()]
    support = int(np.count_nonzero(np.abs(current.theta) > active_tolerance))
    for target in range(support):
        if abs(current.theta[target]) > active_tolerance:
            continue
        candidates = np.flatnonzero(np.abs(current.theta[target + 1 :]) > active_tolerance)
        if len(candidates) == 0:
            raise RuntimeError("Active-atom packing lost a coefficient.")
        source = int(target + 1 + candidates[0])

        duplicate = current.copy()
        duplicate.W[target] = current.W[source]
        _append_if_changed(nodes, duplicate)
        current = duplicate

        transfer = current.copy()
        transfer.theta[target] = current.theta[source]
        transfer.theta[source] = 0.0
        _append_if_changed(nodes, transfer)
        current = transfer

        reset = current.copy()
        reset.W[source] = 0.0
        _append_if_changed(nodes, reset)
        current = reset

    reset_inactive = current.copy()
    reset_inactive.W[support:] = 0.0
    _append_if_changed(nodes, reset_inactive)
    return nodes


def _endpoint_to_reference_path(
    endpoint: ModelState,
    reference: ModelState,
    *,
    active_tolerance: float,
) -> tuple[list[ModelState], CompressionResult]:
    if endpoint.input_dim != reference.input_dim:
        raise ValueError("Endpoint and reference input dimensions differ.")
    if not 1 <= reference.width < endpoint.width:
        raise ValueError("Reference width must satisfy 1 <= l < m.")
    compression = compress_by_cluster_merge(
        endpoint,
        reserve_neurons=reference.width,
        active_tolerance=active_tolerance,
    )
    nodes = [endpoint.copy()]
    _append_if_changed(nodes, compression.state.copy())
    packed_nodes = _pack_active_atoms(
        nodes[-1], active_tolerance=active_tolerance
    )
    for node in packed_nodes[1:]:
        _append_if_changed(nodes, node)

    combined = nodes[-1].copy()
    combined.W[-reference.width :] = reference.W
    _append_if_changed(nodes, combined)

    reference_embedding = combined.copy()
    reference_embedding.theta[:] = 0.0
    reference_embedding.theta[-reference.width :] = reference.theta
    _append_if_changed(nodes, reference_embedding)

    canonical = reference_embedding.copy()
    canonical.W[: -reference.width] = 0.0
    _append_if_changed(nodes, canonical)
    return nodes, compression


def construct_proof_path(
    left: ModelState,
    right: ModelState,
    reference: ModelState,
    *,
    active_tolerance: float = 1e-12,
) -> ProofPathConstruction:
    """Instantiate the compression-and-common-reference path from the proof.

    The reference is a numerical proxy for an l-neuron minimizer.  Optimality
    of that proxy is *not* needed for the returned path to be valid; it affects
    only how small its observed barrier is.
    """
    left_nodes, left_compression = _endpoint_to_reference_path(
        left, reference, active_tolerance=active_tolerance
    )
    right_nodes, right_compression = _endpoint_to_reference_path(
        right, reference, active_tolerance=active_tolerance
    )
    nodes = left_nodes + list(reversed(right_nodes))[1:]
    return ProofPathConstruction(
        nodes=nodes,
        left_compression=left_compression,
        right_compression=right_compression,
        bridge_width=reference.width,
    )


def run_certified_dss(
    objective: EmpiricalObjective,
    a: ModelState,
    b: ModelState,
    *,
    threshold: float,
    config: DSSConfig,
    trainer: Callable[[EmpiricalObjective, ModelState, TrainConfig], object] = train_projected_adam,
) -> PathResult:
    """Build a DSS path and attach a finite-grid error certificate.

    ``certified_upper`` is an analytic upper bound for the maximum objective on
    the returned piecewise-linear path.  It is therefore an upper bound on the
    infimal pair barrier, while ``sampled_max`` is only an observed maximum on
    that particular path.  Neither quantity estimates a lower bound for the
    infimum over all paths.
    """
    endpoint_max = max(objective.value(a), objective.value(b))
    if endpoint_max > threshold + config.certificate_tolerance:
        raise ValueError("Both endpoints must belong to the requested sublevel.")
    records: list[SegmentRecord] = []
    trained_midpoints = 0

    def build(left: ModelState, right: ModelState, depth: int) -> list[ModelState]:
        nonlocal trained_midpoints
        evaluation = _adaptive_evaluate(objective, left, right, threshold, config)
        if evaluation.certified_upper <= threshold + config.certificate_tolerance:
            records.append(SegmentRecord(depth, evaluation, "certified"))
            return [left, right]
        if depth >= config.max_depth:
            records.append(SegmentRecord(depth, evaluation, "depth_limit"))
            return [left, right]

        if evaluation.sampled_max > threshold + config.certificate_tolerance:
            target_config = replace(config.midpoint_train, target_objective=threshold)
            trained = trainer(objective, evaluation.peak, target_config)
            if trained.objective > threshold + config.certificate_tolerance:
                records.append(SegmentRecord(depth, evaluation, "midpoint_training_failed"))
                return [left, right]
            midpoint = trained.state
            trained_midpoints += 1
            status = "trained_split"
        else:
            # No sampled violation: bisect only to reduce the rigorous mesh
            # uncertainty.  This does not invent a trained low-loss midpoint.
            midpoint = interpolate(left, right, 0.5)
            if objective.value(midpoint) > threshold + config.certificate_tolerance:
                records.append(SegmentRecord(depth, evaluation, "uncertain_midpoint"))
                return [left, right]
            status = "resolution_split"

        records.append(SegmentRecord(depth, evaluation, status))
        first = build(left, midpoint, depth + 1)
        second = build(midpoint, right, depth + 1)
        return first[:-1] + second

    nodes = build(a, b, 0)
    sampled_max, certified_upper = evaluate_piecewise_path(
        objective,
        nodes,
        grid_points=max(config.final_grid_points, config.initial_grid_points),
    )
    unresolved = sum(
        item.status in {"depth_limit", "midpoint_training_failed", "uncertain_midpoint"}
        for item in records
    )
    return PathResult(
        nodes=nodes,
        sampled_max=float(sampled_max),
        certified_upper=float(certified_upper),
        certificate_width=float(certified_upper - sampled_max),
        threshold=float(threshold),
        sampled_gap=float(max(0.0, sampled_max - threshold)),
        certified_gap_upper=float(max(0.0, certified_upper - threshold)),
        certified_below_level=bool(
            certified_upper <= threshold + config.certificate_tolerance
        ),
        unresolved_segments=int(unresolved),
        trained_midpoints=int(trained_midpoints),
        records=records,
    )
