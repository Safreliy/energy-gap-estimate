"""Theory-aligned experiments for shallow ReLU barrier estimates."""

from .core import (
    EmpiricalObjective,
    ModelState,
    TrainConfig,
    TrainResult,
    compress_by_cluster_merge,
    compress_by_nearest_cluster_merge,
    make_ridge_teacher_dataset,
    project_first_layer,
    sphericalize_state,
    train_projected_adam,
)
from .paths import (
    DSSConfig,
    PathResult,
    construct_proof_path,
    evaluate_proof_path,
    evaluate_piecewise_path,
    evaluate_segment,
    run_certified_dss,
)

__all__ = [
    "DSSConfig",
    "EmpiricalObjective",
    "ModelState",
    "PathResult",
    "TrainConfig",
    "TrainResult",
    "compress_by_cluster_merge",
    "compress_by_nearest_cluster_merge",
    "construct_proof_path",
    "evaluate_piecewise_path",
    "evaluate_proof_path",
    "evaluate_segment",
    "make_ridge_teacher_dataset",
    "project_first_layer",
    "sphericalize_state",
    "run_certified_dss",
    "train_projected_adam",
]
