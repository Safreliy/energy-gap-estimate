"""Create the manuscript figure/table for the v2 compression diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _groups(rows: list[dict[str, float]]):
    grouped: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["dimension"]), int(row["q"]))].append(row)
    return grouped


def _band(ax, x, values, *, color, label, marker="o"):
    means = np.asarray([np.mean(value) for value in values])
    lows = np.asarray([np.min(value) for value in values])
    highs = np.asarray([np.max(value) for value in values])
    ax.plot(x, means, marker=marker, color=color, label=label, linewidth=1.7)
    ax.fill_between(x, lows, highs, color=color, alpha=0.14, linewidth=0)
    return means


def analyze(result_dir: Path, article_dir: Path) -> dict:
    rows = _read_rows(result_dir / "self_sparsification_cases.csv")
    report = json.loads(
        (result_dir / "self_sparsification_report.json").read_text(encoding="utf-8")
    )
    grouped = _groups(rows)
    dimensions = sorted({int(row["dimension"]) for row in rows})
    qs = sorted({int(row["q"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(dimensions)))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))
    for dimension, color in zip(dimensions, colors):
        case_groups = [grouped[(dimension, q)] for q in qs]
        empirical = [[row["mc_mean_excess"] for row in group] for group in case_groups]
        exact = [[row["exact_expected_excess"] for row in group] for group in case_groups]
        q_scaled = [[row["q_scaled_exact"] for row in group] for group in case_groups]
        ratios = [[row["exact_to_bound_ratio"] for row in group] for group in case_groups]

        empirical_means = _band(
            axes[0], qs, empirical, color=color, label=rf"$n={dimension}$"
        )
        exact_means = np.asarray([np.mean(value) for value in exact])
        axes[0].plot(qs, exact_means, color=color, linestyle="--", linewidth=1.1)
        _band(axes[1], qs, q_scaled, color=color, label=rf"$n={dimension}$")
        _band(axes[2], qs, ratios, color=color, label=rf"$n={dimension}$")

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sampled support $q$")
    axes[0].set_ylabel("mean half-squared-loss objective excess")
    axes[0].set_title("(a) Monte Carlo (solid), exact (dashed)")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("sampled support $q$")
    axes[1].set_ylabel(r"$q\,E[\Delta]/(a^2D_X^2)$")
    axes[1].set_title("(b) dimension-normalized constant")

    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("sampled support $q$")
    axes[2].set_ylabel("exact expectation / analytic bound")
    axes[2].set_title("(c) certificate utilization")
    axes[2].axhline(1.0, color="black", linewidth=1.0, linestyle="-.")
    axes[2].set_ylim(bottom=0.0)

    for ax in axes:
        ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    fig.tight_layout(w_pad=1.35)
    figure_dir = article_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "self_sparsification.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "self_sparsification.png", bbox_inches="tight")
    plt.close(fig)

    summaries: list[dict[str, object]] = []
    for dimension in dimensions:
        q_means = np.asarray(
            [
                np.mean(
                    [row["exact_expected_excess"] for row in grouped[(dimension, q)]]
                )
                for q in qs
            ]
        )
        slope = float(np.polyfit(np.log(np.asarray(qs)), np.log(q_means), 1)[0])
        selected = [row for row in rows if int(row["dimension"]) == dimension]
        summaries.append(
            {
                "dimension": dimension,
                "slope": slope,
                "max_ratio": max(row["exact_to_bound_ratio"] for row in selected),
                "q_scaled_min": min(row["q_scaled_exact"] for row in selected),
                "q_scaled_max": max(row["q_scaled_exact"] for row in selected),
                "max_mc_relative_error": max(
                    abs(row["mc_mean_excess"] - row["exact_expected_excess"])
                    / max(row["exact_expected_excess"], 1e-300)
                    for row in selected
                ),
            }
        )

    table_lines = [
        r"\begin{table}[t]",
        r"\caption{Fixed-dictionary self-sparsification diagnostic. The slope is fitted only as an implementation check; the exact conditional expectation is algebraically proportional to $q^{-1}$. Ranges are over eight dense endpoints and all sampled supports.}",
        r"\label{tab:self-sparsification}",
        r"\centering",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$n$ & fitted slope & max. bound ratio & range of $q\mathbb E\Delta/(a^2D_X^2)$ \\",
        r"\midrule",
    ]
    for item in summaries:
        table_lines.append(
            f"{item['dimension']} & {item['slope']:.3f} & {item['max_ratio']:.3f} & "
            f"[{item['q_scaled_min']:.3f}, {item['q_scaled_max']:.3f}] \\\\"
        )
    table_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (article_dir / "generated_self_sparsification_table.tex").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )

    summary = {
        "source_report": report,
        "dimension_summaries": summaries,
        "max_mc_relative_error": max(
            item["max_mc_relative_error"] for item in summaries
        ),
    }
    (result_dir / "self_sparsification_analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("v2/results/self_sparsification_gpu"),
    )
    parser.add_argument("--article-dir", type=Path, default=Path("v2/article"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.result_dir, args.article_dir), indent=2))


if __name__ == "__main__":
    main()
