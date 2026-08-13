from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .analysis import rate_rows, summarize_models, summarize_pairs, write_csv
from .core import (
    EmpiricalObjective,
    TrainConfig,
    compress_by_cluster_merge,
    initialize_state,
    make_ridge_teacher_dataset,
    train_projected_adam,
)
from .paths import (
    DSSConfig,
    construct_proof_path,
    evaluate_piecewise_path,
    evaluate_proof_path,
    evaluate_segment,
    run_certified_dss,
)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "seed",
        "dimensions",
        "widths",
        "replicates",
        "models_per_width",
        "pairs_per_width",
        "dataset",
        "objective",
        "training",
        "dss",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if 2 * int(config["pairs_per_width"]) > int(config["models_per_width"]):
        raise ValueError("Disjoint pairs require models_per_width >= 2*pairs_per_width.")
    return config


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _train_config(raw: dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        beta1=float(raw.get("beta1", 0.9)),
        beta2=float(raw.get("beta2", 0.999)),
        adam_eps=float(raw.get("adam_eps", 1e-8)),
        patience=int(raw.get("patience", 250)),
        min_delta=float(raw.get("min_delta", 1e-9)),
        log_every=int(raw.get("log_every", 0)),
    )


def _dss_config(raw: dict[str, Any], training: TrainConfig) -> DSSConfig:
    midpoint_raw = dict(asdict(training))
    midpoint_raw.update(raw.get("midpoint_training", {}))
    return DSSConfig(
        max_depth=int(raw["max_depth"]),
        initial_grid_points=int(raw["initial_grid_points"]),
        max_grid_points=int(raw["max_grid_points"]),
        final_grid_points=int(raw["final_grid_points"]),
        certificate_tolerance=float(raw.get("certificate_tolerance", 1e-5)),
        midpoint_train=_train_config(midpoint_raw),
    )


def _train_many(
    objective: EmpiricalObjective,
    initial_states: list[Any],
    training: TrainConfig,
    raw_training: dict[str, Any],
) -> list[Any]:
    backend = str(raw_training.get("backend", "numpy")).lower()
    if backend == "numpy":
        return [train_projected_adam(objective, state, training) for state in initial_states]
    if backend == "torch":
        from .torch_backend import train_projected_adam_batch_torch

        return train_projected_adam_batch_torch(
            objective,
            initial_states,
            training,
            device=str(raw_training.get("device", "cuda")),
            dtype=str(raw_training.get("dtype", "float32")),
        )
    raise ValueError("training.backend must be 'numpy' or 'torch'.")


def _disjoint_pairs(
    eligible: list[int],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    shuffled = np.asarray(eligible, dtype=int).copy()
    rng.shuffle(shuffled)
    usable = min(count, len(shuffled) // 2)
    return [
        (int(shuffled[2 * index]), int(shuffled[2 * index + 1]))
        for index in range(usable)
    ]


def run(config: dict[str, Any], output_dir: Path) -> None:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_seed = int(config["seed"])
    widths = [int(value) for value in config["widths"]]
    dimensions = [int(value) for value in config["dimensions"]]
    replicates = int(config["replicates"])
    models_per_width = int(config["models_per_width"])
    pairs_per_width = int(config["pairs_per_width"])
    training_config = _train_config(config["training"])
    raw_training_config = config["training"]
    dss_config = _dss_config(config["dss"], training_config)
    dataset_config = config["dataset"]
    objective_config = config["objective"]
    moving_slack_fraction = float(config.get("moving_level_slack_fraction_of_zero", 0.02))
    moving_slack_exponent = float(config.get("moving_level_slack_exponent", 0.5))
    level_quantile = float(config.get("level_selection_quantile", 0.75))
    reserve_neurons = int(config.get("compression_reserve_neurons", 2))
    bridge_widths = sorted(
        {int(value) for value in config.get("proof_bridge_widths", [reserve_neurons])}
    )
    fixed_bridge_width = int(config.get("fixed_proof_bridge_width", bridge_widths[0]))
    if fixed_bridge_width not in bridge_widths:
        raise ValueError("fixed_proof_bridge_width must occur in proof_bridge_widths.")
    bridge_starts = int(config.get("proof_bridge_starts", 4))
    active_tolerance = float(config.get("active_tolerance", 1e-10))
    artifact_pairs_per_group = int(config.get("artifact_pairs_per_group", 1))

    model_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    artifacts: dict[str, np.ndarray] = {}
    states: dict[tuple[int, int, int], list[Any]] = {}

    for dimension in dimensions:
        X, y, dataset_metadata = make_ridge_teacher_dataset(
            input_dim=dimension,
            n_samples=int(dataset_config["n_samples"]),
            teacher_width=int(dataset_config["teacher_width"]),
            coefficient_decay=float(dataset_config["coefficient_decay"]),
            seed=master_seed + 100_003 * dimension,
            target_scale=float(dataset_config.get("target_scale", 1.0)),
            target_mode=str(dataset_config.get("target_mode", "regression")),
        )
        objective = EmpiricalObjective(
            X,
            y,
            loss=str(objective_config["loss"]),
            kappa=float(objective_config["kappa"]),
            huber_delta=float(objective_config.get("huber_delta", 0.25)),
        )
        zero_state = initialize_state(1, dimension, np.random.default_rng(0))
        zero_state.theta[:] = 0.0
        zero_objective = objective.value(zero_state)
        artifacts[f"d{dimension}_X"] = X
        artifacts[f"d{dimension}_y"] = y

        for replicate in range(replicates):
            for width in widths:
                initial_pool: list[Any] = []
                seeds: list[int] = []
                for model_index in range(models_per_width):
                    seed = (
                        master_seed
                        + 1_000_003 * dimension
                        + 10_007 * replicate
                        + 101 * width
                        + model_index
                    )
                    initial_pool.append(
                        initialize_state(width, dimension, np.random.default_rng(seed))
                    )
                    seeds.append(seed)
                trained_pool = _train_many(
                    objective, initial_pool, training_config, raw_training_config
                )
                pool: list[Any] = []
                for model_index, (seed, trained) in enumerate(zip(seeds, trained_pool)):
                    pool.append(trained.state)
                    key = f"d{dimension}_r{replicate}_w{width}_m{model_index}"
                    artifacts[f"{key}_W"] = trained.state.W
                    artifacts[f"{key}_theta"] = trained.state.theta
                    model_rows.append(
                        {
                            "dataset": dataset_metadata["type"],
                            "input_dim": dimension,
                            "replicate": replicate,
                            "width": width,
                            "model_index": model_index,
                            "seed": seed,
                            "objective": trained.objective,
                            "data_loss": trained.data_loss,
                            "l1_penalty": trained.l1_penalty,
                            "theta_l1": float(np.sum(np.abs(trained.state.theta))),
                            "active_support": int(
                                np.count_nonzero(np.abs(trained.state.theta) > active_tolerance)
                            ),
                            "max_first_layer_norm": float(
                                np.max(np.linalg.norm(trained.state.W, axis=1))
                            ),
                            "epochs_run": trained.epochs_run,
                            "stop_reason": trained.stop_reason,
                            "zero_objective": zero_objective,
                        }
                    )
                states[(dimension, replicate, width)] = pool
                print(
                    f"[pool] n={dimension} replicate={replicate + 1}/{replicates} "
                    f"width={width} best={min(objective.value(state) for state in pool):.8f}",
                    flush=True,
                )

                # Compression is evaluated on every independently trained endpoint.
                if width > reserve_neurons:
                    for model_index, state in enumerate(pool):
                        before = objective.value(state)
                        compressed = compress_by_cluster_merge(
                            state,
                            reserve_neurons=reserve_neurons,
                            active_tolerance=active_tolerance,
                        )
                        after = objective.value(compressed.state)
                        exact_scale = (
                            objective.loss_lipschitz
                            * objective.d_x
                            * float(np.sum(np.abs(state.theta)))
                        )
                        lemma_bound = (
                            4.0
                            * np.sqrt(dimension)
                            * exact_scale
                            * ((reserve_neurons + 1) / width) ** (1.0 / dimension)
                        )
                        compression_rows.append(
                            {
                                "dataset": dataset_metadata["type"],
                                "input_dim": dimension,
                                "replicate": replicate,
                                "width": width,
                                "model_index": model_index,
                                "reserve_neurons": reserve_neurons,
                                "support_before": compressed.support_before,
                                "support_after": compressed.support_after,
                                "cluster_size": len(compressed.merged_indices),
                                "cluster_diameter": compressed.cluster_diameter,
                                "objective_before": before,
                                "objective_after": after,
                                "objective_increase": after - before,
                                "positive_increase": max(0.0, after - before),
                                "lemma_upper_increment": lemma_bound,
                                "bound_respected": bool(after - before <= lemma_bound + 1e-10),
                            }
                        )

        # A pooled optimization proxy defines lambda_m only after all independent
        # starts are available.  It is never called the true value e(m).
        dimension_models = [row for row in model_rows if row["input_dim"] == dimension]
        pooled_proxy: dict[int, float] = {}
        running_proxy = float("inf")
        for width in sorted(widths):
            width_best = min(
                float(row["objective"])
                for row in dimension_models
                if int(row["width"]) == width
            )
            # A narrower predictor embeds exactly into every larger width by
            # appending zero output coefficients.  The numerical proxy should
            # preserve the same monotonicity as e(m), even when optimization of
            # a wider random initialization is worse.
            running_proxy = min(running_proxy, width_best)
            pooled_proxy[width] = running_proxy
        # Pre-specified balanced sublevels: for each width use the largest
        # within-replicate order statistic needed to retain both the requested
        # quantile and enough endpoints for disjoint pairs.  This makes the
        # threshold common across independent blocks and avoids selecting only
        # unusually successful optimizer seeds.  The fixed level is one common
        # threshold across widths; the moving level follows the width-specific
        # order statistic with a vanishing deterministic slack.
        balanced_levels: dict[int, float] = {}
        for width in widths:
            replicate_levels: list[float] = []
            for replicate in range(replicates):
                losses = sorted(
                    objective.value(state)
                    for state in states[(dimension, replicate, width)]
                )
                quantile_index = max(
                    int(np.ceil(level_quantile * len(losses))) - 1,
                    2 * pairs_per_width - 1,
                )
                quantile_index = min(len(losses) - 1, quantile_index)
                replicate_levels.append(float(losses[quantile_index]))
            balanced_levels[width] = max(replicate_levels)
        fixed_level = max(balanced_levels.values())
        moving_levels = {
            width: value
            + moving_slack_fraction * zero_objective * width ** (-moving_slack_exponent)
            for width, value in balanced_levels.items()
        }

        reference_results: dict[int, Any] = {}
        for candidate_width in bridge_widths:
            reference_initials = []
            for start in range(bridge_starts):
                seed = (
                    master_seed
                    + 70_000_027 * dimension
                    + 10_009 * candidate_width
                    + start
                )
                initial = initialize_state(
                    candidate_width, dimension, np.random.default_rng(seed)
                )
                reference_initials.append(initial)
            reference_candidates = _train_many(
                objective,
                reference_initials,
                training_config,
                raw_training_config,
            )
            result = min(reference_candidates, key=lambda item: item.objective)
            reference_results[candidate_width] = result
            artifacts[f"d{dimension}_proof_reference_w{candidate_width}_W"] = result.state.W
            artifacts[f"d{dimension}_proof_reference_w{candidate_width}_theta"] = result.state.theta
            print(
                f"[reference] n={dimension} width={candidate_width} "
                f"objective={result.objective:.8f}",
                flush=True,
            )

        for replicate in range(replicates):
            for width in widths:
                pool = states[(dimension, replicate, width)]
                losses = [objective.value(state) for state in pool]
                for level_type, threshold in (
                    ("fixed", fixed_level),
                    ("moving", moving_levels[width]),
                ):
                    eligible = [
                        index
                        for index, value in enumerate(losses)
                        if value <= threshold + dss_config.certificate_tolerance
                    ]
                    pair_rng = np.random.default_rng(
                        master_seed
                        + 8_000_009 * dimension
                        + 90_001 * replicate
                        + 307 * width
                        + (0 if level_type == "fixed" else 1)
                    )
                    pairs = _disjoint_pairs(eligible, pairs_per_width, pair_rng)
                    for pair_index, (left_index, right_index) in enumerate(pairs):
                        left = pool[left_index]
                        right = pool[right_index]
                        result = run_certified_dss(
                            objective,
                            left,
                            right,
                            threshold=threshold,
                            config=dss_config,
                            trainer=(
                                lambda obj, state, cfg: _train_many(
                                    obj, [state], cfg, raw_training_config
                                )[0]
                            ),
                        )
                        direct = evaluate_segment(
                            objective,
                            left,
                            right,
                            grid_points=dss_config.final_grid_points,
                        )
                        direct_gap = max(0.0, direct.certified_upper - threshold)
                        admissible_bridges = (
                            [fixed_bridge_width]
                            if level_type == "fixed"
                            else [candidate for candidate in bridge_widths if candidate < width]
                        )
                        admissible_bridges = [
                            candidate for candidate in admissible_bridges if candidate < width
                        ]
                        if not admissible_bridges:
                            raise ValueError(
                                f"No proof bridge width is smaller than network width {width}."
                            )
                        proof_candidates = []
                        for candidate_width in admissible_bridges:
                            candidate_result = reference_results[candidate_width]
                            candidate_path = construct_proof_path(
                                left,
                                right,
                                candidate_result.state,
                                active_tolerance=active_tolerance,
                            )
                            upper = evaluate_proof_path(objective, candidate_path)
                            sampled = upper
                            proof_candidates.append(
                                (upper, sampled, candidate_width, candidate_path, candidate_result)
                            )
                        (
                            proof_certified_upper,
                            proof_sampled_max,
                            selected_bridge_width,
                            proof_path,
                            selected_reference_result,
                        ) = min(proof_candidates, key=lambda item: item[0])
                        proof_gap = max(0.0, proof_certified_upper - threshold)
                        path_key = (
                            f"path_d{dimension}_r{replicate}_w{width}_"
                            f"{level_type}_p{pair_index}"
                        )
                        if pair_index < artifact_pairs_per_group:
                            artifacts[f"{path_key}_W"] = np.stack(
                                [node.W for node in result.nodes]
                            )
                            artifacts[f"{path_key}_theta"] = np.stack(
                                [node.theta for node in result.nodes]
                            )
                            artifacts[f"{path_key}_proof_W"] = np.stack(
                                [node.W for node in proof_path.nodes]
                            )
                            artifacts[f"{path_key}_proof_theta"] = np.stack(
                                [node.theta for node in proof_path.nodes]
                            )
                        pair_rows.append(
                            {
                                "dataset": dataset_metadata["type"],
                                "input_dim": dimension,
                                "replicate": replicate,
                                "width": width,
                                "level_type": level_type,
                                "threshold": threshold,
                                "optimization_proxy": pooled_proxy[width],
                                "eligible_models": len(eligible),
                                "pair_index": pair_index,
                                "model_left": left_index,
                                "model_right": right_index,
                                "endpoint_left": losses[left_index],
                                "endpoint_right": losses[right_index],
                                "sampled_path_max": result.sampled_max,
                                "certified_path_upper": result.certified_upper,
                                "certificate_width": result.certificate_width,
                                "sampled_gap": result.sampled_gap,
                                "certified_gap_upper": result.certified_gap_upper,
                                "direct_sampled_max": direct.sampled_max,
                                "direct_certified_upper": direct.certified_upper,
                                "direct_certificate_width": (
                                    direct.certified_upper - direct.sampled_max
                                ),
                                "direct_certified_gap_upper": direct_gap,
                                "proof_node_max": proof_sampled_max,
                                "proof_certified_upper": proof_certified_upper,
                                "proof_certificate_width": (
                                    proof_certified_upper - proof_sampled_max
                                ),
                                "proof_certified_gap_upper": proof_gap,
                                "proof_path_nodes": len(proof_path.nodes),
                                "proof_bridge_width": selected_bridge_width,
                                "proof_reference_objective": selected_reference_result.objective,
                                "certified_below_level": result.certified_below_level,
                                "unresolved_segments": result.unresolved_segments,
                                "trained_midpoints": result.trained_midpoints,
                                "path_nodes": len(result.nodes),
                            }
                        )
                    print(
                        f"[paths] n={dimension} replicate={replicate + 1}/{replicates} "
                        f"width={width} level={level_type} pairs={len(pairs)}",
                        flush=True,
                    )

    model_summary = summarize_models(model_rows)
    pair_summary = summarize_pairs(pair_rows)
    rates = rate_rows(
        model_summary,
        pair_summary,
        n_bootstrap=int(config.get("bootstrap_repetitions", 2000)),
        seed=master_seed,
    )
    if not model_rows:
        raise RuntimeError("No trained model records were produced.")
    if not pair_rows:
        raise RuntimeError("No eligible endpoint pairs were produced.")
    if any(float(row["max_first_layer_norm"]) > 1.0 + 1e-9 for row in model_rows):
        raise RuntimeError("A trained first-layer row left the closed unit ball.")
    if any(not bool(row["bound_respected"]) for row in compression_rows):
        raise RuntimeError("A constructive compression row violated its analytic bound.")
    expected_pairs = (
        len(dimensions) * replicates * len(widths) * 2 * pairs_per_width
    )
    achieved_pairs = len(pair_rows)
    write_csv(output_dir / "model_records.csv", model_rows)
    write_csv(output_dir / "model_summary.csv", model_summary)
    write_csv(output_dir / "pair_records.csv", pair_rows)
    write_csv(output_dir / "pair_summary.csv", pair_summary)
    write_csv(output_dir / "compression_records.csv", compression_rows)
    write_csv(output_dir / "rate_estimates.csv", rates)
    np.savez_compressed(output_dir / "states_and_paths.npz", **artifacts)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": _git_commit(),
        "expected_pair_records": expected_pairs,
        "achieved_pair_records": achieved_pairs,
        "complete_pair_design": achieved_pairs == expected_pairs,
        "interpretation": {
            "optimization_proxy": "best value found; an upper bound on, not an estimate certified equal to, e(m)",
            "certified_path_upper": "upper bound for one returned path, hence an upper bound for that pair's infimal barrier",
            "sampled_pairs": "a finite endpoint sample; it does not establish a uniform bound over the full sublevel",
            "independence": "replicates are independent blocks; disjoint pairs are used within each block",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Wrote theory-aligned experiment to {output_dir}")
    print(
        f"models={len(model_rows)} pairs={len(pair_rows)}/{expected_pairs} "
        f"elapsed={metadata['elapsed_seconds']:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(_read_config(args.config), args.output)


if __name__ == "__main__":
    main()
