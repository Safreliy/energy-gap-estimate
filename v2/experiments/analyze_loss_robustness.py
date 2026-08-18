from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from theory_experiments.analyze_main_results import validate_design


DIMENSIONS = (1, 2, 4)
WIDTHS = (8, 16, 32, 64, 128)
RUNS = {
    "Huber": "reviewer_main_gpu_n{dimension}",
    "binary cross-entropy": "reviewer_cross_entropy_gpu_n{dimension}",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_loss(
    results_root: Path, template: str
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    records = {stem: [] for stem in ("model_records", "pair_records", "compression_records")}
    metadata: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        run_dir = results_root / template.format(dimension=dimension)
        for stem in records:
            records[stem].extend(read_csv(run_dir / f"{stem}.csv"))
        with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            run_metadata = json.load(handle)
        if not run_metadata.get("complete_pair_design", False):
            raise RuntimeError(f"Incomplete pair design in {run_dir}.")
        metadata.append(run_metadata)
    validate_design(
        records["model_records"],
        records["pair_records"],
        records["compression_records"],
    )
    return records, metadata


def best_relative_gap(row: dict[str, str]) -> float:
    best = min(
        float(row["certified_gap_upper"]),
        float(row["proof_certified_gap_upper"]),
    )
    return best / float(row["threshold"])


def replicate_worst(
    rows: list[dict[str, str]], level_type: str
) -> dict[tuple[int, int], np.ndarray]:
    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["level_type"] != level_type:
            continue
        grouped[
            (int(row["input_dim"]), int(row["width"]), int(row["replicate"]))
        ].append(best_relative_gap(row))
    collapsed: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (dimension, width, _replicate), values in grouped.items():
        # Worst of the three disjoint pairs: one conservative replicate-level
        # statistic, with no pair-level pseudoreplication.
        collapsed[(dimension, width)].append(max(values))
    return {key: np.asarray(values) for key, values in collapsed.items()}


def plot(rows_by_loss: dict[str, list[dict[str, str]]], output_base: Path) -> None:
    styles = {
        "Huber": ("#0072B2", "o", "-"),
        "binary cross-entropy": ("#D55E00", "s", "--"),
    }
    summaries = {
        (loss, level): replicate_worst(rows, level)
        for loss, rows in rows_by_loss.items()
        for level in ("fixed", "moving")
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharex=True, sharey=True)
    panel = 0
    for row_index, level in enumerate(("fixed", "moving")):
        for column_index, dimension in enumerate(DIMENSIONS):
            axis = axes[row_index, column_index]
            for loss in rows_by_loss:
                summary = summaries[(loss, level)]
                color, marker, linestyle = styles[loss]
                medians, lows, highs = [], [], []
                for width in WIDTHS:
                    values = summary[(dimension, width)]
                    medians.append(float(np.median(values)))
                    lows.append(float(np.min(values)))
                    highs.append(float(np.max(values)))
                axis.fill_between(
                    WIDTHS, lows, highs, color=color, alpha=0.10, linewidth=0
                )
                axis.plot(
                    WIDTHS,
                    medians,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.7,
                    markersize=5,
                    label=loss,
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("symlog", linthresh=1e-9, linscale=0.7)
            axis.set_xticks(WIDTHS)
            axis.set_xticklabels(WIDTHS)
            axis.grid(True, which="major", color="#d8d8d8", linewidth=0.55)
            axis.grid(True, which="minor", color="#eeeeee", linewidth=0.35)
            axis.set_title(
                f"({chr(97 + panel)})  {level}, $n={dimension}$",
                loc="left",
                fontsize=10.2,
            )
            panel += 1
            if row_index == 1:
                axis.set_xlabel("width $m$")
            if column_index == 0:
                axis.set_ylabel("best path gap / level")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=9.2,
    )
    figure.subplots_adjust(
        left=0.075, right=0.99, bottom=0.09, top=0.90, wspace=0.10, hspace=0.19
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def summarize(rows: list[dict[str, str]], level_type: str) -> dict[str, Any]:
    high_width = [
        row
        for row in rows
        if int(row["width"]) >= 16 and row["level_type"] == level_type
    ]
    best = np.asarray(
        [
            min(float(row["certified_gap_upper"]), float(row["proof_certified_gap_upper"]))
            for row in high_width
        ]
    )
    threshold = np.asarray([float(row["threshold"]) for row in high_width])
    proof = np.asarray([float(row["proof_certified_gap_upper"]) for row in high_width])
    direct = np.asarray([float(row["direct_certified_gap_upper"]) for row in high_width])
    return {
        "high_width_pairs": len(high_width),
        "max_best_gap": float(np.max(best)),
        "max_best_relative_gap": float(np.max(best / threshold)),
        "proof_zero_fraction": float(np.mean(proof <= 1e-12)),
        "median_direct_relative_gap": float(np.median(direct / threshold)),
        "dss_within_tolerance_fraction": float(
            np.mean([row["certified_below_level"].lower() == "true" for row in high_width])
        ),
    }


def write_table(
    path: Path, summaries: dict[str, dict[str, dict[str, Any]]]
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Loss-robustness summary over all dimensions at widths $m\ge16$. Each loss--level row contains 360 recorded pairs. ``Best'' is the smaller certified DSS or constructive upper gap.}",
        r"\label{tab:loss-robustness}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Loss & level & max gap & max/level & proof zero (\%) & DSS tol. (\%) & med. direct/level \\",
        r"\midrule",
    ]
    display_names = {"Huber": "Huber", "binary cross-entropy": "BCE (logits)"}
    for loss, level_rows in summaries.items():
        for level, row in level_rows.items():
            lines.append(
                f"{display_names[loss]} & {level} & {row['max_best_gap']:.3e} & "
                f"{row['max_best_relative_gap']:.3e} & "
                f"{100.0 * row['proof_zero_fraction']:.1f} & "
                f"{100.0 * row['dss_within_tolerance_fraction']:.1f} & "
                f"{row['median_direct_relative_gap']:.3e} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(results_root: Path, analysis_dir: Path, article_dir: Path) -> None:
    loaded = {
        loss: load_loss(results_root, template) for loss, template in RUNS.items()
    }
    rows_by_loss = {
        loss: records["pair_records"] for loss, (records, _metadata) in loaded.items()
    }
    summaries = {
        loss: {
            level: summarize(rows, level) for level in ("fixed", "moving")
        }
        for loss, rows in rows_by_loss.items()
    }
    invariants = {
        loss: validate_design(
            records["model_records"],
            records["pair_records"],
            records["compression_records"],
        )
        for loss, (records, _metadata) in loaded.items()
    }
    run_metadata = {
        loss: [
            {
                "dimension": int(item["config"]["dimensions"][0]),
                "loss": item["config"]["objective"]["loss"],
                "target_mode": item["config"]["dataset"].get("target_mode", "regression"),
                "elapsed_seconds": float(item["elapsed_seconds"]),
                "complete_pair_design": bool(item["complete_pair_design"]),
            }
            for item in metadata
        ]
        for loss, (_records, metadata) in loaded.items()
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "loss_robustness_report.json").write_text(
        json.dumps(
            {"summaries": summaries, "invariants": invariants, "runs": run_metadata},
            indent=2,
        ),
        encoding="utf-8",
    )
    plot(rows_by_loss, article_dir / "figures" / "loss_robustness")
    write_table(article_dir / "generated_loss_robustness_table.tex", summaries)
    print(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--analysis-dir", type=Path, default=Path("results/reviewer_loss_robustness")
    )
    parser.add_argument("--article-dir", type=Path, default=Path("jamc_article"))
    args = parser.parse_args()
    analyze(args.results_root, args.analysis_dir, args.article_dir)


if __name__ == "__main__":
    main()
