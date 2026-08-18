from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_mean(
    rows: list[dict[str, str]],
    *,
    dimension: int,
    value: str,
    level_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["input_dim"]) != dimension:
            continue
        if level_type is not None and row.get("level_type") != level_type:
            continue
        grouped[int(row["width"])].append(float(row[value]))
    widths = np.asarray(sorted(grouped), dtype=float)
    means = np.asarray([np.mean(grouped[int(width)]) for width in widths])
    spread = np.asarray(
        [np.std(grouped[int(width)], ddof=1) if len(grouped[int(width)]) > 1 else 0.0 for width in widths]
    )
    return widths, means, spread


def _safe_log_values(values: np.ndarray) -> np.ndarray:
    positive = values[values > 0]
    floor = float(np.min(positive) / 2.0) if len(positive) else 1e-12
    return np.maximum(values, floor)


def plot(results_dir: Path) -> list[Path]:
    model_rows = _read_csv(results_dir / "model_summary.csv")
    pair_rows = _read_csv(results_dir / "pair_summary.csv")
    compression_rows = _read_csv(results_dir / "compression_records.csv")
    with (results_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    dimensions = [int(value) for value in metadata["config"]["dimensions"]]
    outputs: list[Path] = []

    figure, axes = plt.subplots(
        len(dimensions),
        3,
        figsize=(13.2, 3.7 * len(dimensions)),
        squeeze=False,
    )
    for row_index, dimension in enumerate(dimensions):
        ax_approx, ax_barrier, ax_compression = axes[row_index]

        widths, minima, minima_sd = _group_mean(
            model_rows, dimension=dimension, value="objective_min"
        )
        floor = float(np.min(minima))
        excess = _safe_log_values(minima - floor)
        ax_approx.errorbar(widths, excess, yerr=np.minimum(minima_sd, 0.8 * excess), marker="o", capsize=3)
        ax_approx.set_xscale("log")
        ax_approx.set_yscale("log")
        ax_approx.set_title(f"n={dimension}: optimization proxy excess")
        ax_approx.set_xlabel("width m")
        ax_approx.set_ylabel(r"best found $F_m$ minus widest proxy")

        for level_type, marker, color in (
            ("fixed", "o", "#277da1"),
            ("moving", "s", "#f3722c"),
        ):
            bw, upper, upper_sd = _group_mean(
                pair_rows,
                dimension=dimension,
                value="proof_upper_max",
                level_type=level_type,
            )
            if len(bw):
                upper = _safe_log_values(upper)
                ax_barrier.errorbar(
                    bw,
                    upper,
                    yerr=np.minimum(upper_sd, 0.8 * upper),
                    marker=marker,
                    color=color,
                    capsize=3,
                    label=level_type,
                )
        ax_barrier.set_xscale("log")
        ax_barrier.set_yscale("log")
        ax_barrier.set_title("proof-inspired path upper gaps")
        ax_barrier.set_xlabel("width m")
        ax_barrier.set_ylabel(r"max over sampled pairs")
        ax_barrier.legend()

        grouped: dict[int, list[float]] = defaultdict(list)
        bound_grouped: dict[int, list[float]] = defaultdict(list)
        for row in compression_rows:
            if int(row["input_dim"]) == dimension:
                grouped[int(row["width"])].append(float(row["positive_increase"]))
                bound_grouped[int(row["width"])].append(float(row["lemma_upper_increment"]))
        cw = np.asarray(sorted(grouped), dtype=float)
        if len(cw):
            observed = _safe_log_values(
                np.asarray([max(values) for _, values in sorted(grouped.items())])
            )
            bound = _safe_log_values(
                np.asarray([max(values) for _, values in sorted(bound_grouped.items())])
            )
            ax_compression.plot(cw, observed, marker="o", label="observed increase")
            ax_compression.plot(cw, bound, linestyle="--", label="lemma upper bound")
        ax_compression.set_xscale("log")
        ax_compression.set_yscale("log")
        ax_compression.set_title("constructive cluster merge")
        ax_compression.set_xlabel("width m")
        ax_compression.set_ylabel(r"objective increment")
        ax_compression.legend()

    figure.suptitle(
        "Theory-aligned diagnostics: computation is observation, dashed bound is analytic",
        y=1.01,
    )
    figure.tight_layout()
    main_path = results_dir / "theory_aligned_summary.png"
    figure.savefig(main_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    outputs.append(main_path)

    # Direct interpolation versus DSS provides an algorithm ablation.  The
    # theorem is about existence of a path, not superiority of a path finder.
    figure, axes = plt.subplots(1, len(dimensions), figsize=(5.0 * len(dimensions), 4.0), squeeze=False)
    for index, dimension in enumerate(dimensions):
        ax = axes[0, index]
        for level_type, marker in (("fixed", "o"), ("moving", "s")):
            widths, dss, _ = _group_mean(
                pair_rows,
                dimension=dimension,
                value="path_upper_max",
                level_type=level_type,
            )
            _, direct, _ = _group_mean(
                pair_rows,
                dimension=dimension,
                value="direct_upper_max",
                level_type=level_type,
            )
            _, proof, _ = _group_mean(
                pair_rows,
                dimension=dimension,
                value="proof_upper_max",
                level_type=level_type,
            )
            if len(widths):
                ax.plot(widths, _safe_log_values(dss), marker=marker, label=f"DSS {level_type}")
                ax.plot(widths, _safe_log_values(direct), marker=marker, linestyle="--", label=f"direct {level_type}")
                ax.plot(widths, _safe_log_values(proof), marker=marker, linestyle=":", label=f"proof path {level_type}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"n={dimension}")
        ax.set_xlabel("width m")
        ax.set_ylabel("max certified returned-path gap")
        ax.legend(fontsize=8)
    figure.tight_layout()
    ablation_path = results_dir / "path_algorithm_ablation.png"
    figure.savefig(ablation_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    outputs.append(ablation_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    for path in plot(args.results):
        print(path)


if __name__ == "__main__":
    main()
