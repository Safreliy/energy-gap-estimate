from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import (
    EmpiricalObjective,
    ModelState,
    compress_by_nearest_cluster_merge,
    sphericalize_state,
)


DIMENSIONS = (1, 2, 4)
WIDTHS = (16, 32, 64, 128)
JITTER_GRID = np.linspace(0.0, 0.25, 33)
ACTIVE_TOLERANCE = 1e-14
RESERVE_NEURONS = 4
LEVEL_TOLERANCE = 1e-5


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dense_split(state: ModelState) -> tuple[ModelState, np.ndarray]:
    state = sphericalize_state(state, active_tolerance=ACTIVE_TOLERANCE)
    active = np.flatnonzero(np.abs(state.theta) > ACTIVE_TOLERANCE)
    if len(active) == 0:
        raise RuntimeError("A selected endpoint has no active atom to split.")
    parents = np.resize(active, state.width)
    counts = np.bincount(parents, minlength=state.width)
    dense = ModelState(
        W=state.W[parents].copy(),
        theta=np.asarray(
            [state.theta[parent] / counts[parent] for parent in parents],
            dtype=np.float64,
        ),
    )
    return dense, parents


def _tangent_directions(
    dense: ModelState,
    parents: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    tangent = np.zeros_like(dense.W)
    if dense.input_dim == 1:
        return tangent
    for parent in np.unique(parents):
        indices = np.flatnonzero(parents == parent)
        raw = rng.normal(size=(len(indices), dense.input_dim))
        base = dense.W[indices]
        raw -= np.sum(raw * base, axis=1, keepdims=True) * base
        if len(indices) > 1:
            raw -= np.mean(raw, axis=0, keepdims=True)
            raw -= np.sum(raw * base, axis=1, keepdims=True) * base
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        fallback = np.zeros_like(raw)
        fallback[:, 0] = 1.0
        fallback -= np.sum(fallback * base, axis=1, keepdims=True) * base
        fallback_norms = np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback = fallback / np.maximum(fallback_norms, 1e-15)
        raw = np.where(norms > 1e-12, raw / np.maximum(norms, 1e-15), fallback)
        tangent[indices] = raw
    return tangent


def _jitter(dense: ModelState, tangent: np.ndarray, amplitude: float) -> ModelState:
    if amplitude == 0.0:
        return dense.copy()
    W = dense.W + float(amplitude) * tangent
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-15)
    return ModelState(W=W, theta=dense.theta.copy())


def _select_dense_endpoint(
    objective: EmpiricalObjective,
    state: ModelState,
    threshold: float,
    seed: int,
) -> tuple[ModelState, float, float]:
    dense, parents = _dense_split(state)
    exact_value = objective.value(dense)
    if exact_value > threshold + LEVEL_TOLERANCE:
        raise RuntimeError("Exact splitting left the recorded level beyond its tolerance.")
    if dense.input_dim == 1:
        return dense, 0.0, exact_value
    tangent = _tangent_directions(dense, parents, np.random.default_rng(seed))
    accepted = dense
    accepted_amplitude = 0.0
    accepted_value = exact_value
    for amplitude in JITTER_GRID[1:]:
        candidate = _jitter(dense, tangent, float(amplitude))
        value = objective.value(candidate)
        if value <= threshold + LEVEL_TOLERANCE:
            accepted = candidate
            accepted_amplitude = float(amplitude)
            accepted_value = value
    return accepted, accepted_amplitude, accepted_value


def _fixed_selected_models(pair_rows: list[dict[str, str]]) -> dict[tuple[int, int], tuple[float, list[int]]]:
    groups: dict[tuple[int, int], dict[str, object]] = {}
    for row in pair_rows:
        if row["level_type"] != "fixed" or int(row["width"]) not in WIDTHS:
            continue
        key = (int(row["replicate"]), int(row["width"]))
        group = groups.setdefault(
            key, {"threshold": float(row["threshold"]), "models": set()}
        )
        if abs(float(group["threshold"]) - float(row["threshold"])) > 1e-12:
            raise RuntimeError("Fixed-level threshold changed within a group.")
        models = group["models"]
        assert isinstance(models, set)
        models.update((int(row["model_left"]), int(row["model_right"])))
    result: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for key, group in groups.items():
        models = sorted(group["models"])
        if len(models) != 6:
            raise RuntimeError(f"Expected six disjoint fixed-level endpoints in {key}.")
        result[key] = (float(group["threshold"]), models)
    return result


def run(results_root: Path, analysis_dir: Path, article_dir: Path) -> None:
    records: list[dict[str, object]] = []
    for dimension in DIMENSIONS:
        result_dir = results_root / f"reviewer_main_gpu_n{dimension}"
        pair_rows = _read_csv(result_dir / "pair_records.csv")
        selected = _fixed_selected_models(pair_rows)
        archive = np.load(result_dir / "states_and_paths.npz", allow_pickle=False)
        metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8"))
        objective_config = metadata["config"]["objective"]
        objective = EmpiricalObjective(
            archive[f"d{dimension}_X"],
            archive[f"d{dimension}_y"],
            loss=objective_config["loss"],
            kappa=float(objective_config["kappa"]),
            huber_delta=float(objective_config.get("huber_delta", 0.25)),
        )
        for (replicate, width), (threshold, model_indices) in sorted(selected.items()):
            for model_index in model_indices:
                prefix = f"d{dimension}_r{replicate}_w{width}_m{model_index}"
                original = ModelState(
                    W=archive[f"{prefix}_W"], theta=archive[f"{prefix}_theta"]
                )
                dense, jitter, dense_value = _select_dense_endpoint(
                    objective,
                    original,
                    threshold,
                    seed=20260813
                    + 1_000_000 * dimension
                    + 10_000 * replicate
                    + 100 * width
                    + model_index,
                )
                compression = compress_by_nearest_cluster_merge(
                    dense,
                    reserve_neurons=RESERVE_NEURONS,
                    active_tolerance=ACTIVE_TOLERANCE,
                )
                compressed_value = objective.value(compression.state)
                increment = compressed_value - dense_value
                exact_bound = (
                    objective.loss_lipschitz
                    * objective.d_x
                    * float(np.sum(np.abs(dense.theta)))
                    * compression.cluster_diameter
                )
                respected = increment <= exact_bound + 1e-10
                if not respected:
                    raise RuntimeError("A dense-endpoint merge violated its exact certificate.")
                if compression.support_before != width:
                    raise RuntimeError("Dense splitting did not activate every coordinate.")
                if compression.support_after > width - RESERVE_NEURONS:
                    raise RuntimeError("Dense stress did not free the required coordinates.")
                records.append(
                    {
                        "input_dim": dimension,
                        "replicate": replicate,
                        "width": width,
                        "model_index": model_index,
                        "threshold": threshold,
                        "level_tolerance": LEVEL_TOLERANCE,
                        "original_objective": objective.value(original),
                        "dense_objective": dense_value,
                        "accepted_jitter": jitter,
                        "support_before": compression.support_before,
                        "support_after": compression.support_after,
                        "removed_support": compression.support_before
                        - compression.support_after,
                        "cluster_size": len(compression.merged_indices),
                        "cluster_diameter": compression.cluster_diameter,
                        "compressed_objective": compressed_value,
                        "objective_increment": increment,
                        "positive_increment": max(0.0, increment),
                        "exact_diameter_bound": exact_bound,
                        "increment_to_bound": (
                            max(0.0, increment) / exact_bound if exact_bound > 0.0 else 0.0
                        ),
                        "bound_respected": respected,
                    }
                )

    expected = len(DIMENSIONS) * len(WIDTHS) * 10 * 6
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} stress records, found {len(records)}.")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "dense_endpoint_records.csv", records)

    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["input_dim"]), int(row["width"]))].append(row)
    summaries: list[dict[str, object]] = []
    for dimension in DIMENSIONS:
        for width in WIDTHS:
            rows = grouped[(dimension, width)]
            diameters = np.asarray([float(row["cluster_diameter"]) for row in rows])
            ratios = np.asarray([float(row["increment_to_bound"]) for row in rows])
            increments = np.asarray([float(row["positive_increment"]) for row in rows])
            jitters = np.asarray([float(row["accepted_jitter"]) for row in rows])
            removed = np.asarray([int(row["removed_support"]) for row in rows])
            summaries.append(
                {
                    "input_dim": dimension,
                    "width": width,
                    "records": len(rows),
                    "min_removed_support": int(np.min(removed)),
                    "median_cluster_diameter": float(np.median(diameters)),
                    "max_cluster_diameter": float(np.max(diameters)),
                    "median_positive_increment": float(np.median(increments)),
                    "max_positive_increment": float(np.max(increments)),
                    "max_increment_to_bound": float(np.max(ratios)),
                    "positive_jitter_fraction": float(np.mean(jitters > 0.0)),
                    "median_jitter": float(np.median(jitters)),
                }
            )
    _write_csv(analysis_dir / "dense_endpoint_summary.csv", summaries)

    colors = {1: "#0072B2", 2: "#D55E00", 4: "#009E73"}
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.25), constrained_layout=True)
    for dimension in DIMENSIONS:
        per_width = [grouped[(dimension, width)] for width in WIDTHS]
        diameter_median = [
            np.median([float(row["cluster_diameter"]) for row in rows])
            for rows in per_width
        ]
        diameter_min = [
            np.min([float(row["cluster_diameter"]) for row in rows]) for rows in per_width
        ]
        diameter_max = [
            np.max([float(row["cluster_diameter"]) for row in rows]) for rows in per_width
        ]
        axes[0].plot(WIDTHS, diameter_median, marker="o", color=colors[dimension], label=f"n={dimension}")
        axes[0].fill_between(WIDTHS, diameter_min, diameter_max, color=colors[dimension], alpha=0.13)
        axes[1].plot(
            WIDTHS,
            [max(float(row["positive_increment"]) for row in rows) for rows in per_width],
            marker="o",
            color=colors[dimension],
        )
        axes[2].plot(
            WIDTHS,
            [max(float(row["increment_to_bound"]) for row in rows) for rows in per_width],
            marker="o",
            color=colors[dimension],
        )
    axes[0].set_title("(a) selected cluster geometry", loc="left", fontsize=10.5)
    axes[0].set_ylabel("cluster diameter")
    axes[0].set_yscale("symlog", linthresh=1e-8)
    axes[1].set_title("(b) worst merge increment", loc="left", fontsize=10.5)
    axes[1].set_ylabel(r"max $(F_{\rm merged}-F_{\rm dense})_+$")
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[2].set_title("(c) exact-certificate ratio", loc="left", fontsize=10.5)
    axes[2].set_ylabel(r"max $\Delta F_+/B_{\rm diam}$")
    axes[2].set_yscale("symlog", linthresh=1e-5)
    axes[2].set_ylim(-1e-6, 1.25)
    axes[2].axhline(1.0, color="#333333", linestyle="-.", linewidth=1.0)
    for axis in axes:
        axis.set_xlabel("width m")
        axis.set_xscale("log", base=2)
        axis.set_xticks(WIDTHS)
        axis.set_xticklabels([str(width) for width in WIDTHS])
        axis.grid(alpha=0.22)
    figure.legend(loc="outside upper center", ncol=3, frameon=False)
    for suffix in ("pdf", "png"):
        target = article_dir / "figures" / f"dense_endpoint_stress.{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=240, bbox_inches="tight")
    plt.close(figure)

    report = {
        "scope": "fixed-level selected Huber endpoints at widths m>=16",
        "construction": "exact sphericalization and coefficient splitting, followed by sublevel-calibrated tangent jitter",
        "records": len(records),
        "records_per_dimension_width": 60,
        "all_coordinates_active_before_merge": all(
            int(row["support_before"]) == int(row["width"]) for row in records
        ),
        "minimum_removed_support": min(int(row["removed_support"]) for row in records),
        "maximum_positive_increment": max(float(row["positive_increment"]) for row in records),
        "maximum_increment_to_bound": max(float(row["increment_to_bound"]) for row in records),
        "bound_violations": sum(not bool(row["bound_respected"]) for row in records),
        "positive_jitter_fraction_n2_n4": float(
            np.mean(
                [
                    float(row["accepted_jitter"]) > 0.0
                    for row in records
                    if int(row["input_dim"]) > 1
                ]
            )
        ),
        "warning": "This constructed finite stress test validates the active merge mechanism and its exact inequality; it is not evidence for an asymptotic exponent.",
    }
    (analysis_dir / "dense_endpoint_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("results/dense_endpoint_stress")
    )
    parser.add_argument("--article-dir", type=Path, default=Path("jamc_article"))
    args = parser.parse_args()
    run(args.results_root, args.analysis_dir, args.article_dir)


if __name__ == "__main__":
    main()
