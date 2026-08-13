from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("dataset", "input_dim", "replicate", "width", "level_type")
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items()):
        sampled = np.asarray([float(row["sampled_gap"]) for row in values])
        upper = np.asarray([float(row["certified_gap_upper"]) for row in values])
        direct = np.asarray([float(row["direct_certified_gap_upper"]) for row in values])
        proof = np.asarray([float(row["proof_certified_gap_upper"]) for row in values])
        resolution = np.asarray([float(row["certificate_width"]) for row in values])
        summaries.append(
            {
                **dict(zip(keys, group_key)),
                "threshold": float(values[0]["threshold"]),
                "eligible_models": int(values[0]["eligible_models"]),
                "pairs": len(values),
                "sampled_gap_max": float(np.max(sampled)),
                "sampled_gap_median": float(np.median(sampled)),
                "path_upper_max": float(np.max(upper)),
                "path_upper_q90": float(np.quantile(upper, 0.9)),
                "direct_upper_max": float(np.max(direct)),
                "proof_upper_max": float(np.max(proof)),
                "proof_upper_q90": float(np.quantile(proof, 0.9)),
                "certificate_width_max": float(np.max(resolution)),
                "certified_success_rate": float(
                    np.mean([bool(row["certified_below_level"]) for row in values])
                ),
                "unresolved_segments": int(
                    np.sum([int(row["unresolved_segments"]) for row in values])
                ),
            }
        )
    return summaries


def summarize_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("dataset", "input_dim", "replicate", "width")
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, values in sorted(grouped.items()):
        objective = np.asarray([float(row["objective"]) for row in values])
        summaries.append(
            {
                **dict(zip(keys, group_key)),
                "models": len(values),
                "objective_min": float(np.min(objective)),
                "objective_median": float(np.median(objective)),
                "objective_q90": float(np.quantile(objective, 0.9)),
                "active_support_median": float(
                    np.median([float(row["active_support"]) for row in values])
                ),
                "l1_norm_median": float(
                    np.median([float(row["theta_l1"]) for row in values])
                ),
                "max_first_layer_norm": float(
                    np.max([float(row["max_first_layer_norm"]) for row in values])
                ),
            }
        )
    return summaries


def _slope(widths: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(widths) & np.isfinite(values) & (widths > 0) & (values > 0)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(np.log(widths[mask]), np.log(values[mask]), 1)[0])


def block_bootstrap_slopes(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    n_bootstrap: int,
    seed: int,
    transform: Callable[[dict[int, float]], dict[int, float]] | None = None,
) -> dict[str, float]:
    """Fit a log-log slope and resample independent experiment replicates."""
    by_replicate: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_replicate[int(row["replicate"])][int(row["width"])] = float(row[value_key])
    replicate_ids = np.asarray(sorted(by_replicate), dtype=int)
    widths = np.asarray(sorted({width for values in by_replicate.values() for width in values}))
    if len(replicate_ids) < 2 or len(widths) < 3:
        return {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    def aggregate(ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        values_by_width: dict[int, list[float]] = defaultdict(list)
        for replicate in ids:
            for width, value in by_replicate[int(replicate)].items():
                values_by_width[width].append(value)
        aggregate_values = {
            width: float(np.mean(values)) for width, values in values_by_width.items()
        }
        if transform is not None:
            aggregate_values = transform(aggregate_values)
        used_widths = np.asarray(sorted(aggregate_values), dtype=float)
        used_values = np.asarray([aggregate_values[int(width)] for width in used_widths])
        return used_widths, used_values

    base_widths, base_values = aggregate(replicate_ids)
    base_slope = _slope(base_widths, base_values)
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(replicate_ids, size=len(replicate_ids), replace=True)
        boot_widths, boot_values = aggregate(sampled)
        slope = _slope(boot_widths, boot_values)
        if np.isfinite(slope):
            slopes.append(slope)
    if not slopes:
        return {"slope": base_slope, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "slope": base_slope,
        "ci_low": float(np.quantile(slopes, 0.025)),
        "ci_high": float(np.quantile(slopes, 0.975)),
    }


def rate_rows(
    model_summary: list[dict[str, Any]],
    pair_summary: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    dimensions = sorted({int(row["input_dim"]) for row in model_summary})
    output: list[dict[str, Any]] = []
    for dimension in dimensions:
        model_rows = [row for row in model_summary if int(row["input_dim"]) == dimension]

        def approximation_excess(values: dict[int, float]) -> dict[int, float]:
            monotone: dict[int, float] = {}
            running = float("inf")
            for width in sorted(values):
                running = min(running, values[width])
                monotone[width] = running
            floor = min(monotone.values())
            scale = max(abs(floor), 1.0)
            epsilon = 1e-10 * scale
            return {
                width: max(value - floor, epsilon) for width, value in monotone.items()
            }

        approximation = block_bootstrap_slopes(
            model_rows,
            value_key="objective_min",
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 * dimension,
            transform=approximation_excess,
        )
        proxy_by_width: dict[int, list[float]] = defaultdict(list)
        for row in model_rows:
            proxy_by_width[int(row["width"])].append(float(row["objective_min"]))
        proxy_means = {width: float(np.mean(values)) for width, values in proxy_by_width.items()}
        running = float("inf")
        proxy_excess: list[float] = []
        floor = min(proxy_means.values())
        for width in sorted(proxy_means):
            running = min(running, proxy_means[width])
            proxy_excess.append(max(0.0, running - floor))
        proxy_zero_fraction = float(np.mean(np.asarray(proxy_excess) <= 1e-12))
        output.append(
            {
                "input_dim": dimension,
                "quantity": "optimization_proxy_excess",
                **(
                    {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
                    if proxy_zero_fraction >= 0.5
                    else approximation
                ),
                "zero_fraction": proxy_zero_fraction,
                "slope_status": (
                    "not_estimable_due_to_floor_or_zeros"
                    if proxy_zero_fraction >= 0.5
                    else "descriptive"
                ),
                "theory_reference": "conditional; fitted slope is not an estimate of true s",
            }
        )
        for level_type in ("fixed", "moving"):
            selected = [
                row
                for row in pair_summary
                if int(row["input_dim"]) == dimension and row["level_type"] == level_type
            ]
            if not selected:
                continue
            estimate = block_bootstrap_slopes(
                selected,
                value_key="path_upper_max",
                n_bootstrap=n_bootstrap,
                seed=seed + 2000 * dimension + (0 if level_type == "fixed" else 1),
            )
            observed_values = np.asarray(
                [float(row["path_upper_max"]) for row in selected], dtype=float
            )
            zero_fraction = float(np.mean(observed_values <= 1e-12))
            if zero_fraction >= 0.5:
                estimate = {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
            output.append(
                {
                    "input_dim": dimension,
                    "quantity": f"{level_type}_sampled_pair_path_upper",
                    **estimate,
                    "zero_fraction": zero_fraction,
                    "slope_status": (
                        "not_estimable_due_to_zeros"
                        if zero_fraction >= 0.5
                        else "descriptive"
                    ),
                    "theory_reference": (
                        f"worst-case theorem slope >= {-1.0 / dimension:.6g}"
                        if level_type == "fixed"
                        else "compare with -s/(n s + 1) only after specifying s"
                    ),
                }
            )
            proof_estimate = block_bootstrap_slopes(
                selected,
                value_key="proof_upper_max",
                n_bootstrap=n_bootstrap,
                seed=seed + 3000 * dimension + (0 if level_type == "fixed" else 1),
            )
            proof_values = np.asarray(
                [float(row["proof_upper_max"]) for row in selected], dtype=float
            )
            proof_zero_fraction = float(np.mean(proof_values <= 1e-12))
            if proof_zero_fraction >= 0.5:
                proof_estimate = {
                    "slope": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                }
            output.append(
                {
                    "input_dim": dimension,
                    "quantity": f"{level_type}_proof_path_upper",
                    **proof_estimate,
                    "zero_fraction": proof_zero_fraction,
                    "slope_status": (
                        "not_estimable_due_to_zeros"
                        if proof_zero_fraction >= 0.5
                        else "descriptive"
                    ),
                    "theory_reference": (
                        f"constructive analogue; worst-case theorem slope >= {-1.0 / dimension:.6g}"
                        if level_type == "fixed"
                        else "constructive analogue at moving empirical levels"
                    ),
                }
            )
    return output
