"""GPU/CPU diagnostic matched to the v2 self-sparsification theorem.

The source endpoint uses a dense width-M fixed dictionary.  We compress it to
q atoms by the exact multinomial construction, compare Monte Carlo averages
with the closed-form expectation for half-squared loss, and check the balanced
all-weight-decay lift.  This is a mechanism and implementation diagnostic; it
does not sample an entire sublevel and is not a barrier lower bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ridge_endpoint(torch, features, target, ridge: float, atomic_mass: float):
    n = features.shape[0]
    gram = features.T @ features / n
    rhs = features.T @ target / n
    theta = torch.linalg.solve(
        gram + ridge * torch.eye(features.shape[1], device=features.device, dtype=features.dtype),
        rhs,
    )
    mass = torch.sum(torch.abs(theta))
    if float(mass) == 0.0:
        raise RuntimeError("The ridge endpoint has zero atomic mass.")
    return theta * (atomic_mass / mass)


def _run_case(torch, *, dimension: int, endpoint: int, config: dict, device: str):
    seed = int(config["seed"]) + 100003 * dimension + 997 * endpoint
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    dtype = getattr(torch, config.get("dtype", "float64"))
    n_samples = int(config["n_samples"])
    source_width = int(config["source_width"])
    teacher_width = int(config["teacher_width"])
    atomic_mass = float(config["atomic_mass"])
    ridge = float(config["ridge"])
    kappa = float(config["kappa"])
    trials = int(config["trials"])

    X = torch.randn((n_samples, dimension), generator=generator, device=device, dtype=dtype)
    teacher_W = torch.randn(
        (teacher_width, dimension), generator=generator, device=device, dtype=dtype
    )
    teacher_W /= torch.linalg.vector_norm(teacher_W, dim=1, keepdim=True).clamp_min(1e-15)
    ranks = torch.arange(1, teacher_width + 1, device=device, dtype=dtype)
    teacher_theta = torch.where(
        torch.rand(teacher_width, generator=generator, device=device) < 0.5,
        -torch.ones(teacher_width, device=device, dtype=dtype),
        torch.ones(teacher_width, device=device, dtype=dtype),
    ) / ranks.pow(float(config["teacher_decay"]))
    teacher_theta /= torch.sum(torch.abs(teacher_theta))
    y = torch.relu(X @ teacher_W.T) @ teacher_theta
    noise_scale = float(config.get("noise_scale", 0.0))
    if noise_scale:
        y = y + noise_scale * torch.randn(
            y.shape, generator=generator, device=device, dtype=dtype
        )

    W = torch.randn(
        (source_width, dimension), generator=generator, device=device, dtype=dtype
    )
    W /= torch.linalg.vector_norm(W, dim=1, keepdim=True).clamp_min(1e-15)
    features = torch.relu(X @ W.T)
    theta = _ridge_endpoint(torch, features, y, ridge, atomic_mass)
    prediction = features @ theta
    residual = prediction - y
    base_risk = 0.5 * torch.mean(residual.square())
    base_objective = base_risk + kappa * torch.sum(torch.abs(theta))
    covariance = X.T @ X / n_samples
    d_x_squared = torch.linalg.eigvalsh(covariance)[-1]
    probabilities = torch.abs(theta) / torch.sum(torch.abs(theta))
    signs = torch.sign(theta)

    raw_radii = torch.sqrt(torch.abs(theta))
    raw_theta = signs * raw_radii
    raw_W = W * raw_radii[:, None]
    raw_prediction = torch.relu(X @ raw_W.T) @ raw_theta
    atomic_penalty = kappa * torch.sum(torch.abs(theta))
    raw_penalty = 0.5 * kappa * (
        torch.sum(raw_theta.square()) + torch.sum(raw_W.square())
    )

    rows: list[dict[str, object]] = []
    for q in [int(value) for value in config["sample_widths"]]:
        if q > source_width:
            continue
        indices = torch.multinomial(
            probabilities,
            num_samples=trials * q,
            replacement=True,
            generator=generator,
        ).reshape(trials, q)
        counts = torch.zeros((trials, source_width), device=device, dtype=dtype)
        counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=dtype))
        sampled_theta = counts * signs[None, :] * (atomic_mass / q)
        sampled_prediction = sampled_theta @ features.T
        errors = sampled_prediction - prediction[None, :]
        sampled_risk = 0.5 * torch.mean(
            (sampled_prediction - y[None, :]).square(), dim=1
        )
        risk_excess = sampled_risk - base_risk
        linear_terms = torch.mean(residual[None, :] * errors, dim=1)
        remainders = 0.5 * torch.mean(errors.square(), dim=1)

        # Exact expectation over the atom sampler for this finite distribution.
        second_atom_moment = atomic_mass * torch.mean(
            features.square() @ torch.abs(theta)
        )
        predictor_second_moment = torch.mean(prediction.square())
        exact_remainder = 0.5 * torch.clamp(
            second_atom_moment - predictor_second_moment, min=0.0
        ) / q
        smooth_bound = 0.5 * atomic_mass**2 * d_x_squared / q
        sampled_masses = torch.sum(torch.abs(sampled_theta), dim=1)
        sampled_supports = torch.count_nonzero(sampled_theta, dim=1)

        rows.append(
            {
                "dimension": dimension,
                "endpoint": endpoint,
                "source_width": source_width,
                "q": q,
                "trials": trials,
                "atomic_mass": atomic_mass,
                "d_x_squared": float(d_x_squared),
                "base_risk": float(base_risk),
                "base_objective": float(base_objective),
                "exact_expected_excess": float(exact_remainder),
                "mc_mean_excess": float(torch.mean(risk_excess)),
                "mc_excess_se": float(torch.std(risk_excess, correction=1) / trials**0.5),
                "mc_mean_linear_term": float(torch.mean(linear_terms)),
                "mc_mean_remainder": float(torch.mean(remainders)),
                "smooth_bound": float(smooth_bound),
                "exact_to_bound_ratio": float(exact_remainder / smooth_bound),
                "q_scaled_exact": float(q * exact_remainder / (atomic_mass**2 * d_x_squared)),
                "best_objective_excess": float(torch.min(risk_excess)),
                "max_mass_error": float(torch.max(torch.abs(sampled_masses - atomic_mass))),
                "max_support_excess": int(torch.max(sampled_supports).item() - q),
                "balance_prediction_error": float(torch.max(torch.abs(raw_prediction - prediction))),
                "balance_penalty_error": float(torch.abs(raw_penalty - atomic_penalty)),
            }
        )
    return rows


def run(config: dict, output_dir: Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("This diagnostic requires PyTorch.") from exc

    requested = str(config.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not config.get("allow_cpu_fallback", False):
            raise RuntimeError("CUDA was requested but is unavailable.")
        requested = "cpu"
    started = time.time()
    rows: list[dict[str, object]] = []
    for dimension in [int(value) for value in config["dimensions"]]:
        for endpoint in range(int(config["endpoints_per_dimension"])):
            rows.extend(
                _run_case(
                    torch,
                    dimension=dimension,
                    endpoint=endpoint,
                    config=config,
                    device=requested,
                )
            )
            print(
                f"[self-sparsification] n={dimension} "
                f"endpoint={endpoint + 1}/{config['endpoints_per_dimension']}",
                flush=True,
            )

    _write_csv(output_dir / "self_sparsification_cases.csv", rows)
    report = {
        "status": "OBS mechanism diagnostic; not a sublevel supremum or barrier lower bound",
        "config": config,
        "device": requested,
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if requested.startswith("cuda")
            else None
        ),
        "platform": platform.platform(),
        "elapsed_seconds": time.time() - started,
        "records": len(rows),
        "max_mass_error": max(float(row["max_mass_error"]) for row in rows),
        "max_support_excess": max(int(row["max_support_excess"]) for row in rows),
        "max_exact_to_bound_ratio": max(
            float(row["exact_to_bound_ratio"]) for row in rows
        ),
        "max_balance_prediction_error": max(
            float(row["balance_prediction_error"]) for row in rows
        ),
        "max_balance_penalty_error": max(
            float(row["balance_penalty_error"]) for row in rows
        ),
        "q_scaled_exact_range": [
            min(float(row["q_scaled_exact"]) for row in rows),
            max(float(row["q_scaled_exact"]) for row in rows),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "self_sparsification_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "self_sparsification_gpu.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("v2/results/self_sparsification_gpu")
    )
    args = parser.parse_args()
    print(json.dumps(run(_load_config(args.config), args.output_dir), indent=2))


if __name__ == "__main__":
    main()
