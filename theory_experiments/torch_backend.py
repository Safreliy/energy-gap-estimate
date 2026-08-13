from __future__ import annotations

from typing import Sequence

import numpy as np

from .core import (
    EmpiricalObjective,
    ModelState,
    TrainConfig,
    TrainResult,
    project_first_layer,
)


def train_projected_adam_batch_torch(
    objective: EmpiricalObjective,
    initial_states: Sequence[ModelState],
    config: TrainConfig,
    *,
    device: str = "cuda",
    dtype: str = "float32",
) -> list[TrainResult]:
    """Batched independent projected-proximal Adam trajectories in PyTorch.

    The leading tensor dimension indexes independent models.  There is no
    coupling in either the objective or the update, so batching changes only
    execution, not the individual optimization recurrences.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on GPU host
        raise RuntimeError("The torch backend requires PyTorch.") from exc

    if not initial_states:
        return []
    first_shape = initial_states[0].W.shape
    for state in initial_states:
        objective.validate_state(state)
        if state.W.shape != first_shape:
            raise ValueError("All states in a CUDA batch must have one shape.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    if dtype not in dtype_map:
        raise ValueError("torch dtype must be 'float32' or 'float64'.")
    torch_dtype = dtype_map[dtype]

    X = torch.as_tensor(objective.X, dtype=torch_dtype, device=device)
    y = torch.as_tensor(objective.y, dtype=torch_dtype, device=device)
    W = torch.as_tensor(
        np.stack([state.W for state in initial_states]),
        dtype=torch_dtype,
        device=device,
    ).clone()
    theta = torch.as_tensor(
        np.stack([state.theta for state in initial_states]),
        dtype=torch_dtype,
        device=device,
    ).clone()
    m_W = torch.zeros_like(W)
    v_W = torch.zeros_like(W)
    m_theta = torch.zeros_like(theta)
    v_theta = torch.zeros_like(theta)
    batch_size = len(initial_states)
    best_values = torch.full((batch_size,), float("inf"), dtype=torch_dtype, device=device)
    best_data = torch.zeros(batch_size, dtype=torch_dtype, device=device)
    best_l1 = torch.zeros(batch_size, dtype=torch_dtype, device=device)
    best_W = W.clone()
    best_theta = theta.clone()
    stale = torch.zeros(batch_size, dtype=torch.int64, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    epochs_run = torch.zeros(batch_size, dtype=torch.int64, device=device)
    histories: list[list[float]] = [[] for _ in initial_states]
    stop_reasons = ["max_epochs" for _ in initial_states]

    def components() -> tuple[object, object, object, object, object, object]:
        preactivation = torch.einsum("nd,bmd->bnm", X, W)
        activation = torch.relu(preactivation)
        prediction = torch.einsum("bnm,bm->bn", activation, theta)
        if objective.loss == "huber":
            residual = prediction - y[None, :]
            absolute = torch.abs(residual)
            data_values = torch.where(
                absolute <= objective.huber_delta,
                0.5 * residual.square(),
                objective.huber_delta * (absolute - 0.5 * objective.huber_delta),
            )
            derivative = torch.clamp(
                residual, -objective.huber_delta, objective.huber_delta
            )
        else:
            data_values = torch.nn.functional.softplus(prediction) - y[None, :] * prediction
            derivative = torch.sigmoid(prediction) - y[None, :]
        data_loss = torch.mean(data_values, dim=1)
        l1_penalty = objective.kappa * torch.sum(torch.abs(theta), dim=1)
        total = data_loss + l1_penalty
        return total, data_loss, l1_penalty, preactivation, activation, derivative

    with torch.no_grad():
        total, data_loss, l1_penalty, _, _, _ = components()
        best_values.copy_(total)
        best_data.copy_(data_loss)
        best_l1.copy_(l1_penalty)
        for index, value in enumerate(total.detach().cpu().tolist()):
            histories[index].append(float(value))

        for epoch in range(1, config.epochs + 1):
            total, data_loss, l1_penalty, preactivation, activation, derivative = components()
            derivative = derivative / len(objective.X)
            grad_theta = torch.einsum("bnm,bn->bm", activation, derivative)
            gate = preactivation > 0.0
            weighted = derivative[:, :, None] * theta[:, None, :] * gate
            grad_W = torch.einsum("bnm,nd->bmd", weighted, X)
            mask_W = active[:, None, None]
            mask_theta = active[:, None]
            grad_W = torch.where(mask_W, grad_W, torch.zeros_like(grad_W))
            grad_theta = torch.where(mask_theta, grad_theta, torch.zeros_like(grad_theta))

            m_W.mul_(config.beta1).add_(grad_W, alpha=1.0 - config.beta1)
            v_W.mul_(config.beta2).addcmul_(grad_W, grad_W, value=1.0 - config.beta2)
            m_theta.mul_(config.beta1).add_(grad_theta, alpha=1.0 - config.beta1)
            v_theta.mul_(config.beta2).addcmul_(
                grad_theta, grad_theta, value=1.0 - config.beta2
            )
            correction1 = 1.0 - config.beta1**epoch
            correction2 = 1.0 - config.beta2**epoch
            W_step = config.learning_rate * (m_W / correction1) / (
                torch.sqrt(v_W / correction2) + config.adam_eps
            )
            W.sub_(torch.where(mask_W, W_step, torch.zeros_like(W_step)))
            norms = torch.linalg.vector_norm(W, dim=2, keepdim=True)
            W.div_(torch.clamp(norms, min=1.0))

            theta_metric = torch.sqrt(v_theta / correction2) + config.adam_eps
            theta_step = config.learning_rate * (m_theta / correction1) / theta_metric
            theta.sub_(torch.where(mask_theta, theta_step, torch.zeros_like(theta_step)))
            shrink = config.learning_rate * objective.kappa / theta_metric
            prox = torch.sign(theta) * torch.clamp(torch.abs(theta) - shrink, min=0.0)
            theta.copy_(torch.where(mask_theta, prox, theta))
            epochs_run[active] = epoch

            total, data_loss, l1_penalty, _, _, _ = components()
            improved = active & (total < best_values - config.min_delta)
            if torch.any(improved):
                best_values[improved] = total[improved]
                best_data[improved] = data_loss[improved]
                best_l1[improved] = l1_penalty[improved]
                best_W[improved] = W[improved]
                best_theta[improved] = theta[improved]
            stale = torch.where(improved, torch.zeros_like(stale), stale + active.to(stale.dtype))
            for index, value in enumerate(total.detach().cpu().tolist()):
                histories[index].append(float(value))

            if config.target_objective is not None:
                reached = active & (best_values <= config.target_objective)
                for index in torch.nonzero(reached, as_tuple=False).flatten().detach().cpu().tolist():
                    stop_reasons[int(index)] = "target"
                active[reached] = False
            if config.patience > 0:
                exhausted = active & (stale >= config.patience)
                for index in torch.nonzero(exhausted, as_tuple=False).flatten().detach().cpu().tolist():
                    stop_reasons[int(index)] = "patience"
                active[exhausted] = False
            if not torch.any(active):
                break
            if config.log_every and epoch % config.log_every == 0:
                print(
                    f"cuda epoch={epoch} active={int(active.sum().item())} "
                    f"best={float(best_values.min().item()):.8f}",
                    flush=True,
                )

    W_cpu = best_W.detach().cpu().double().numpy()
    theta_cpu = best_theta.detach().cpu().double().numpy()
    values_cpu = best_values.detach().cpu().double().numpy()
    data_cpu = best_data.detach().cpu().double().numpy()
    l1_cpu = best_l1.detach().cpu().double().numpy()
    epochs_cpu = epochs_run.detach().cpu().numpy()
    results: list[TrainResult] = []
    for index in range(batch_size):
        # Re-evaluate in float64 NumPy so every reported objective shares the
        # exact same evaluator, irrespective of the optimization backend.  A
        # float32 vector projected onto the sphere can acquire a norm slightly
        # above one after conversion to float64, so enforce feasibility once
        # more in the evaluator's precision before validating the state.
        state = ModelState(project_first_layer(W_cpu[index]), theta_cpu[index])
        value, data_value, l1_value = objective.components(state)
        results.append(
            TrainResult(
                state=state,
                objective=value,
                data_loss=data_value,
                l1_penalty=l1_value,
                epochs_run=int(epochs_cpu[index]),
                stop_reason=stop_reasons[index],
                history=histories[index],
            )
        )
    return results
