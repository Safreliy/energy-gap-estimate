from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DIMENSIONS = (1, 2, 4)
WIDTHS = (8, 16, 32, 64, 128)
LEVELS = ("fixed", "moving")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_records(results_root: Path, stem: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dimension in DIMENSIONS:
        rows.extend(
            read_csv(
                results_root
                / f"reviewer_main_gpu_n{dimension}"
                / f"{stem}.csv"
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values(
    rows: Iterable[dict[str, str]], key: str, *, dimension: int | None = None
) -> np.ndarray:
    selected = [
        float(row[key])
        for row in rows
        if dimension is None or int(row["input_dim"]) == dimension
    ]
    return np.asarray(selected, dtype=float)


def gap_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return ratios without warnings; an exact zero denominator means infinity."""
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.inf, dtype=float),
        where=denominator > 0.0,
    )


def validate_design(
    model_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    compression_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if len(model_rows) != 1200:
        raise RuntimeError(f"Expected 1200 models, found {len(model_rows)}.")
    if len(pair_rows) != 900:
        raise RuntimeError(f"Expected 900 pair rows, found {len(pair_rows)}.")
    if len(compression_rows) != 1200:
        raise RuntimeError(
            f"Expected 1200 compression rows, found {len(compression_rows)}."
        )

    grouped_pairs: dict[tuple[int, int, int, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in pair_rows:
        grouped_pairs[
            (
                int(row["input_dim"]),
                int(row["replicate"]),
                int(row["width"]),
                row["level_type"],
            )
        ].append(row)
    if len(grouped_pairs) != 300 or any(len(group) != 3 for group in grouped_pairs.values()):
        raise RuntimeError("The pair design is not balanced at three pairs per group.")
    for key, group in grouped_pairs.items():
        endpoints = [
            int(row[column])
            for row in group
            for column in ("model_left", "model_right")
        ]
        if len(set(endpoints)) != len(endpoints):
            raise RuntimeError(f"Endpoint reuse within pair group {key}.")

    max_first_layer_norm = max(float(row["max_first_layer_norm"]) for row in model_rows)
    if max_first_layer_norm > 1.0 + 1e-9:
        raise RuntimeError("A trained first-layer row left the unit ball.")
    compression_violations = sum(
        row["bound_respected"].strip().lower() != "true" for row in compression_rows
    )
    if compression_violations:
        raise RuntimeError("A compression record violates the analytic bound.")

    threshold_violations = sum(
        max(float(row["endpoint_left"]), float(row["endpoint_right"]))
        > float(row["threshold"]) + 1e-5
        for row in pair_rows
    )
    if threshold_violations:
        raise RuntimeError("A selected endpoint lies outside its numerical level.")
    return {
        "models": len(model_rows),
        "pairs": len(pair_rows),
        "pair_groups": len(grouped_pairs),
        "compression_records": len(compression_rows),
        "max_first_layer_norm": max_first_layer_norm,
        "compression_bound_violations": compression_violations,
        "endpoint_threshold_violations": threshold_violations,
    }


def dimension_summary(
    model_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    compression_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        dim_models = [row for row in model_rows if int(row["input_dim"]) == dimension]
        proxy: dict[int, float] = {}
        running = float("inf")
        for width in WIDTHS:
            running = min(
                running,
                min(
                    float(row["objective"])
                    for row in dim_models
                    if int(row["width"]) == width
                ),
            )
            proxy[width] = running

        high_width = [
            row
            for row in pair_rows
            if int(row["input_dim"]) == dimension and int(row["width"]) >= 16
        ]
        dss = np.asarray([float(row["certified_gap_upper"]) for row in high_width])
        proof = np.asarray(
            [float(row["proof_certified_gap_upper"]) for row in high_width]
        )
        direct = np.asarray(
            [float(row["direct_certified_gap_upper"]) for row in high_width]
        )
        thresholds = np.asarray([float(row["threshold"]) for row in high_width])
        best = np.minimum(dss, proof)
        compression = [
            row
            for row in compression_rows
            if int(row["input_dim"]) == dimension
        ]
        ratios = np.asarray(
            [
                float(row["positive_increase"])
                / float(row["lemma_upper_increment"])
                for row in compression
            ]
        )
        output.append(
            {
                "input_dim": dimension,
                "optimization_proxy_m8": proxy[8],
                "optimization_proxy_m128": proxy[128],
                "proxy_relative_decrease": (proxy[8] - proxy[128]) / proxy[8],
                "high_width_pairs": len(high_width),
                "proof_zero_gap_fraction": float(np.mean(proof <= 1e-12)),
                "best_path_max_gap": float(np.max(best)),
                "best_path_max_relative_gap": float(np.max(best / thresholds)),
                "dss_median_gap": float(np.median(dss)),
                "direct_median_gap": float(np.median(direct)),
                "median_direct_to_dss_ratio": float(np.median(gap_ratio(direct, dss))),
                "max_compression_to_bound_ratio": float(np.max(ratios)),
            }
        )
    return output


def replicate_maxima(
    pair_rows: list[dict[str, str]], value_key: str
) -> dict[tuple[int, str, int], np.ndarray]:
    grouped: dict[tuple[int, str, int, int], list[float]] = defaultdict(list)
    for row in pair_rows:
        grouped[
            (
                int(row["input_dim"]),
                row["level_type"],
                int(row["width"]),
                int(row["replicate"]),
            )
        ].append(float(row[value_key]))
    output: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for (dimension, level, width, _replicate), group_values in grouped.items():
        output[(dimension, level, width)].append(max(group_values))
    return {key: np.asarray(group, dtype=float) for key, group in output.items()}


def set_gap_axis(axis: Any) -> None:
    axis.set_yscale("symlog", linthresh=1e-9, linscale=0.7)
    axis.set_ylim(-2e-10, 2.1e-1)
    axis.set_yticks([0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1])
    axis.set_yticklabels(["0", "$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{-1}$"])
    axis.set_xscale("log", base=2)
    axis.set_xticks(WIDTHS)
    axis.set_xticklabels([str(width) for width in WIDTHS])
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.55)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.35)


def plot_path_diagnostics(pair_rows: list[dict[str, str]], output_base: Path) -> None:
    methods = (
        ("certified DSS", "certified_gap_upper", "#0072B2", "o", "-"),
        ("proof-inspired", "proof_certified_gap_upper", "#D55E00", "s", "-"),
        ("direct interpolation", "direct_certified_gap_upper", "#6C757D", "^", "--"),
    )
    summaries = {key: replicate_maxima(pair_rows, key) for _, key, *_ in methods}
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.7), sharey=True)
    letters = "abcdef"
    for row_index, level in enumerate(LEVELS):
        for column_index, dimension in enumerate(DIMENSIONS):
            axis = axes[row_index, column_index]
            for label, key, color, marker, linestyle in methods:
                medians: list[float] = []
                lows: list[float] = []
                highs: list[float] = []
                for width in WIDTHS:
                    group = summaries[key][(dimension, level, width)]
                    medians.append(float(np.median(group)))
                    lows.append(float(np.min(group)))
                    highs.append(float(np.max(group)))
                axis.fill_between(
                    WIDTHS, lows, highs, color=color, alpha=0.09, linewidth=0
                )
                axis.plot(
                    WIDTHS,
                    medians,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    markersize=4.8,
                )
            set_gap_axis(axis)
            axis.set_title(
                f"({letters[row_index * 3 + column_index]})  "
                f"$n={dimension}$, {level} level",
                loc="left",
                fontsize=10.5,
            )
            if row_index == 1:
                axis.set_xlabel("width $m$")
            if column_index == 0:
                axis.set_ylabel("returned-path upper gap")
    handles = [
        Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
               linewidth=1.6, markersize=4.8, label=label)
        for label, _key, color, marker, linestyle in methods
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.91, wspace=0.10, hspace=0.28)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def plot_implementation_checks(
    model_rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    compression_rows: list[dict[str, str]],
    output_base: Path,
) -> None:
    colors = {1: "#009E73", 2: "#0072B2", 4: "#CC79A7"}
    markers = {1: "o", 2: "s", 4: "^"}
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.65))

    axis = axes[0]
    for dimension in DIMENSIONS:
        dim_rows = [row for row in model_rows if int(row["input_dim"]) == dimension]
        proxy: list[float] = []
        running = float("inf")
        for width in WIDTHS:
            running = min(
                running,
                min(
                    float(row["objective"])
                    for row in dim_rows
                    if int(row["width"]) == width
                ),
            )
            proxy.append(running)
        normalized = np.asarray(proxy) / proxy[0]
        axis.plot(
            WIDTHS,
            normalized,
            color=colors[dimension],
            marker=markers[dimension],
            label=f"$n={dimension}$",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(WIDTHS)
    axis.set_xticklabels(WIDTHS)
    axis.set_xlabel("width $m$")
    axis.set_ylabel("pooled proxy / proxy at $m=8$")
    axis.set_title("(a) Optimization proxy", loc="left", fontsize=10.5)
    axis.grid(True, color="#dedede", linewidth=0.55)

    axis = axes[1]
    for dimension in DIMENSIONS:
        ratios: list[float] = []
        for width in WIDTHS:
            group = [
                row
                for row in compression_rows
                if int(row["input_dim"]) == dimension
                and int(row["width"]) == width
            ]
            ratios.append(
                max(
                    float(row["positive_increase"])
                    / float(row["lemma_upper_increment"])
                    for row in group
                )
            )
        axis.plot(
            WIDTHS,
            ratios,
            color=colors[dimension],
            marker=markers[dimension],
            label=f"$n={dimension}$",
        )
    axis.axhline(1.0, color="#333333", linestyle="-.", linewidth=1.1, label="analytic limit")
    axis.set_xscale("log", base=2)
    axis.set_xticks(WIDTHS)
    axis.set_xticklabels(WIDTHS)
    axis.set_ylim(-0.005, 1.05)
    axis.set_xlabel("width $m$")
    axis.set_ylabel("max observed increment / bound")
    axis.set_title("(b) Cluster-merge check", loc="left", fontsize=10.5)
    axis.grid(True, color="#dedede", linewidth=0.55)

    axis = axes[2]
    linestyles = {"fixed": "-", "moving": "--"}
    for dimension in DIMENSIONS:
        for level in LEVELS:
            rates: list[float] = []
            for width in WIDTHS:
                group = [
                    row
                    for row in pair_rows
                    if int(row["input_dim"]) == dimension
                    and int(row["width"]) == width
                    and row["level_type"] == level
                ]
                rates.append(
                    100.0
                    * np.mean(
                        [row["certified_below_level"].lower() == "true" for row in group]
                    )
                )
            axis.plot(
                WIDTHS,
                rates,
                color=colors[dimension],
                marker=markers[dimension],
                linestyle=linestyles[level],
                label=f"$n={dimension}$, {level}",
            )
    axis.set_xscale("log", base=2)
    axis.set_xticks(WIDTHS)
    axis.set_xticklabels(WIDTHS)
    axis.set_ylim(-3, 103)
    axis.set_xlabel("width $m$")
    axis.set_ylabel(r"DSS within $10^{-5}$ tolerance (\%)")
    axis.set_title("(c) DSS certificate status", loc="left", fontsize=10.5)
    axis.grid(True, color="#dedede", linewidth=0.55)
    legend_handles = [
        Line2D([0], [0], color=colors[dimension], marker=markers[dimension],
               linestyle="-", label=f"$n={dimension}$")
        for dimension in DIMENSIONS
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="-", label="fixed level"),
            Line2D([0], [0], color="#333333", linestyle="--", label="moving level"),
            Line2D([0], [0], color="#333333", linestyle="-.", linewidth=1.1,
                   label="analytic limit in (b)"),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=8.4,
        columnspacing=1.15,
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.82, wspace=0.28)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Frozen main-run summary. All path quantities are upper gaps for the recorded endpoint pairs, not estimates of the uniform sublevel barrier. The three path columns use only $m\ge16$; the compression ratio uses all widths. ``Best path'' means the smaller of the certified DSS and exact constructive upper gaps; $B$ is the analytic cluster-merge increment bound.}",
        r"\label{tab:main-computation}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        r"$n$ & $\widehat e(8)$ & $\widehat e(128)$ & zero constructive & max best-path & median direct/DSS & max $\Delta_{\rm cmp}/B$ \\",
        r" & & & gap (\%) & gap & ratio & \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['input_dim']} & "
            f"{row['optimization_proxy_m8']:.6f} & "
            f"{row['optimization_proxy_m128']:.6f} & "
            f"{100.0 * row['proof_zero_gap_fraction']:.1f} & "
            f"{row['best_path_max_gap']:.3e} & "
            f"{row['median_direct_to_dss_ratio']:.2e} & "
            f"{row['max_compression_to_bound_ratio']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(results_root: Path, analysis_dir: Path, article_dir: Path) -> None:
    model_rows = load_records(results_root, "model_records")
    pair_rows = load_records(results_root, "pair_records")
    compression_rows = load_records(results_root, "compression_records")
    invariants = validate_design(model_rows, pair_rows, compression_rows)
    summaries = dimension_summary(model_rows, pair_rows, compression_rows)

    high_width_pairs = [row for row in pair_rows if int(row["width"]) >= 16]
    dss = values(high_width_pairs, "certified_gap_upper")
    proof = values(high_width_pairs, "proof_certified_gap_upper")
    direct = values(high_width_pairs, "direct_certified_gap_upper")
    threshold = values(high_width_pairs, "threshold")
    best = np.minimum(dss, proof)
    global_summary = {
        "high_width_pairs": len(high_width_pairs),
        "best_path_zero_fraction": float(np.mean(best <= 1e-12)),
        "best_path_max_gap": float(np.max(best)),
        "best_path_max_relative_gap": float(np.max(best / threshold)),
        "dss_median_gap": float(np.median(dss)),
        "direct_median_gap": float(np.median(direct)),
        "direct_gap_min": float(np.min(direct)),
        "direct_gap_max": float(np.max(direct)),
        "median_direct_to_dss_ratio": float(np.median(gap_ratio(direct, dss))),
        "dss_within_tolerance_fraction": float(
            np.mean(
                [
                    row["certified_below_level"].strip().lower() == "true"
                    for row in high_width_pairs
                ]
            )
        ),
        "unresolved_row_fraction": float(
            np.mean([int(row["unresolved_segments"]) > 0 for row in high_width_pairs])
        ),
    }
    report = {
        "evidence_status": {
            "scope": "finite seeded distribution and recorded endpoint pairs",
            "uniform_sublevel_claim": False,
            "slope_claim": False,
            "reason_slopes_are_not_central": (
                "finite-range fits are dominated by the m=8 to m=16 transition "
                "and a roughly 1e-5 DSS certificate floor thereafter"
            ),
        },
        "invariants": invariants,
        "global_high_width_summary": global_summary,
        "by_dimension": summaries,
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "numerical_summary.csv", summaries)
    (analysis_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_latex_table(article_dir / "generated_main_results_table.tex", summaries)
    figure_dir = article_dir / "figures"
    plot_path_diagnostics(pair_rows, figure_dir / "width_path_diagnostics")
    plot_implementation_checks(
        model_rows,
        pair_rows,
        compression_rows,
        figure_dir / "implementation_checks",
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("results/reviewer_main_analysis")
    )
    parser.add_argument("--article-dir", type=Path, default=Path("jamc_article"))
    args = parser.parse_args()
    analyze(args.results_root, args.analysis_dir, args.article_dir)


if __name__ == "__main__":
    main()
