import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import random
import time
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd
from dataclasses import dataclass, asdict
import math
from torch.func import functional_call

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleNNClassification(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        super(SimpleNNClassification, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def normalize_first_layer_fast(model: nn.Module) -> None:
    if not hasattr(model, "fc1") or not hasattr(model, "fc2"):
        return
    fc1 = model.fc1
    fc2 = model.fc2
    if not hasattr(fc1, "weight") or fc1.weight is None:
        return
    with torch.no_grad():
        weights = fc1.weight.data
        norms = torch.norm(weights, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-12)
        weights.div_(norms)
        if fc1.bias is not None:
            fc1.bias.data.div_(norms.squeeze(1))
        if hasattr(fc2, "weight") and fc2.weight is not None:
            fc2.weight.data.mul_(norms.t())


def get_l1_params(model):
    param_layers = []
    for module in model.modules():
        if list(module.parameters(recurse=False)):
            param_layers.append(module)
    if len(param_layers) < 2:
        raise ValueError("Model must have at least two parameterized layers.")
    second_layer = param_layers[1]
    l1_params = list(second_layer.parameters())
    return l1_params


def get_l1_param_names(model):
    param_layers = []
    for name, module in model.named_modules():
        if list(module.parameters(recurse=False)):
            param_layers.append((name, module))
    if len(param_layers) < 2:
        raise ValueError("Model must have at least two parameterized layers.")
    layer_name, layer = param_layers[1]
    names = []
    for param_name, _ in layer.named_parameters(recurse=False):
        if layer_name:
            names.append(f"{layer_name}.{param_name}")
        else:
            names.append(param_name)
    return names


def max_loss_along_path(
    model: nn.Module,
    loss_func,
    theta_a: List[torch.Tensor],
    theta_b: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    steps: int = 100,
    l1_lambda: float = 0.01,
) -> Tuple[float, float]:
    max_loss, t_max = -float("inf"), 0.0

    data = [(x.to(device), y.to(device)) for x, y in dataloader]

    theta_a = [p.to(device) for p in theta_a]
    theta_b = [p.to(device) for p in theta_b]
    param_names = [name for name, _ in model.named_parameters()]
    theta_a_dict = {name: p for name, p in zip(param_names, theta_a)}
    theta_b_dict = {name: p for name, p in zip(param_names, theta_b)}
    l1_param_names = get_l1_param_names(model)

    with torch.no_grad():
        for t in torch.linspace(0, 1, steps):
            t_scalar = t.item()
            # Create interpolated parameters
            interp_params = {
                name: (1 - t_scalar) * theta_a_dict[name]
                + t_scalar * theta_b_dict[name]
                for name in param_names
            }

            total = 0.0
            for x, y in data:
                output = functional_call(model, interp_params, x)
                total += loss_func(output, y).item()
            avg_loss = total / len(data)

            if l1_lambda != 0:
                l1_reg = 0.0
                for name in l1_param_names:
                    if name in interp_params:
                        l1_reg += interp_params[name].abs().sum().item()
                avg_loss += l1_lambda * l1_reg

            if avg_loss > max_loss:
                max_loss, t_max = avg_loss, t_scalar

    return max_loss, t_max

def train_model_to_threshold(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    lr: float = 0.01,
    threshold: float = 0.1,
    max_epochs: int = 150,
    l1_lambda: float = 0.01,
    normalize_first_layer_weights: bool = False,
) -> None:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9)
    criterion = nn.MSELoss()

    l1_params = get_l1_params(model)

    for epoch in range(max_epochs):
        total = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            # L1 regularization
            l1_reg = sum(torch.sum(torch.abs(p)) for p in l1_params)
            loss += l1_lambda * l1_reg

            loss.backward()
            optimizer.step()
            if normalize_first_layer_weights:
                normalize_first_layer_fast(model)
            total += loss.item()
        avg_loss = total / len(dataloader)
        if avg_loss < threshold:
            return
        scheduler.step(avg_loss)
    # print(f"Reached loss: {avg_loss:.4f}")

def compute_avg_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion,
    l1_lambda: float = 0.01,
) -> float:
    model = model.to(device)
    model_was_training = model.training
    model.eval()

    total_loss = 0.0
    num_batches = 0

    l1_params = get_l1_params(model)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            loss = criterion(preds, y)

            l1_reg = sum(torch.sum(torch.abs(p)) for p in l1_params)
            loss = loss + l1_lambda * l1_reg

            total_loss += loss.item()
            num_batches += 1

    if model_was_training:
        model.train()

    return total_loss / max(1, num_batches)


def compute_avg_loss_for_params(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_func,
    params: List[torch.Tensor],
    l1_lambda: float = 0.01,
) -> float:
    model = model.to(device)
    data = [(x.to(device), y.to(device)) for x, y in dataloader]
    param_names = [name for name, _ in model.named_parameters()]
    params = [p.to(device) for p in params]
    param_dict = {name: p for name, p in zip(param_names, params)}
    l1_param_names = get_l1_param_names(model)

    total_loss = 0.0
    with torch.no_grad():
        for x, y in data:
            output = functional_call(model, param_dict, x)
            loss = loss_func(output, y)
            total_loss += loss.item()

    avg_loss = total_loss / max(1, len(data))

    if l1_lambda != 0:
        l1_reg = 0.0
        for name in l1_param_names:
            if name in param_dict:
                l1_reg += param_dict[name].abs().sum().item()
        avg_loss += l1_lambda * l1_reg

    return avg_loss

def train_model_to_threshold_classifier(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    lr: float = 0.01,
    threshold: float = 0.1,
    max_epochs: int = 100,
    l1_lambda: float = 0.01,
    normalize_first_layer_weights: bool = False,
) -> None:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9)
    criterion = nn.BCEWithLogitsLoss()

    l1_params = get_l1_params(model)

    for epoch in range(max_epochs):
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)

            # L1 regularization
            l1_reg = sum(torch.sum(torch.abs(p)) for p in l1_params)
            loss += l1_lambda * l1_reg

            loss.backward()
            optimizer.step()
            if normalize_first_layer_weights:
                normalize_first_layer_fast(model)

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if avg_loss < threshold:
            return
        scheduler.step(avg_loss)
    #print(f"Reached loss: {avg_loss:.4f}")


# Dynamic String Sampling
def dss(
    cls,
    base_model: nn.Module,
    loss_func,
    trainer,
    theta_a: List[torch.Tensor],
    theta_b: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    threshold: float,
    max_depth: int = 10,
    depth: int = 0,
    losses: List | None = None,
    energy_gaps: List[float] | None = None,
    barrier_losses: List[float] | None = None,
    hit_max_depth: List[bool] | None = None,
    l1_lambda=0.01,
    max_epochs=100,
    path_steps: int = 100,
    normalize_first_layer_weights: bool = False,
    **kwargs,
) -> List[List[torch.Tensor]]:
    if losses is None:
        losses = []
    base_model = base_model.to(device)
    theta_a = [p.to(device) for p in theta_a]
    theta_b = [p.to(device) for p in theta_b]
    loss, t = max_loss_along_path(
        base_model,
        loss_func,
        theta_a,
        theta_b,
        dataloader,
        steps=path_steps,
        l1_lambda=l1_lambda,
    )
    if losses is not None:
        losses.append(loss)
    if loss > threshold and barrier_losses is not None:
        barrier_losses.append(loss)

    if depth > max_depth:
        #print(
        #    f"🟥 Iteration {len(losses) + 1} reached {depth} depth with loss {losses[-1]:.4f}"
        #)
        if hit_max_depth is not None:
            hit_max_depth.append(True)
        if energy_gaps is not None:
            energy_gaps.append(max(0.0, loss - threshold))
        theta_mid = [p_a * (1 - t) + p_b * t for p_a, p_b in zip(theta_a, theta_b)]
        return [theta_a, theta_mid, theta_b]

    if loss <= threshold:
        #print(
        #    f"🟩 Iteration {len(losses) + 1} reached {depth} depth with loss {losses[-1]:.4f}"
        #)
        return [theta_a, theta_b]

    # Create and train midpoint
    theta_mid = [p_a * (1 - t) + p_b * t for p_a, p_b in zip(theta_a, theta_b)]
    mid_model = cls(**kwargs).to(device)
    mid_model.load_state_dict(base_model.state_dict())
    with torch.no_grad():
        for param, val in zip(mid_model.parameters(), theta_mid):
            param.copy_(val)
    trainer(
        mid_model,
        dataloader,
        max_epochs=max_epochs,
        threshold=threshold,
        l1_lambda=l1_lambda,
        normalize_first_layer_weights=normalize_first_layer_weights,
    )
    theta_mid_trained = [p.detach().clone() for p in mid_model.parameters()]
    # Recursive path building
    left = dss(
        cls,
        base_model,
        loss_func,
        trainer,
        theta_a,
        theta_mid_trained,
        dataloader,
        threshold,
        max_depth,
        depth + 1,
        losses,
        energy_gaps,
        barrier_losses,
        hit_max_depth,
        l1_lambda=l1_lambda,
        max_epochs=max_epochs,
        path_steps=path_steps,
        normalize_first_layer_weights=normalize_first_layer_weights,
        **kwargs,
    )
    right = dss(
        cls,
        base_model,
        loss_func,
        trainer,
        theta_mid_trained,
        theta_b,
        dataloader,
        threshold,
        max_depth,
        depth + 1,
        losses,
        energy_gaps,
        barrier_losses,
        hit_max_depth,
        l1_lambda=l1_lambda,
        max_epochs=max_epochs,
        path_steps=path_steps,
        normalize_first_layer_weights=normalize_first_layer_weights,
        **kwargs,
    )

    return left[:-1] + right


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.flatten() for p in params])


def unflatten_params(
    flat_tensor: torch.Tensor, param_shapes: List[torch.Size]
) -> List[torch.Tensor]:
    params = []
    current = 0
    for shape in param_shapes:
        size = torch.prod(torch.tensor(shape)).item()
        tensor = flat_tensor[current : current + size].view(shape)
        params.append(tensor)
        current += size
    return params


def generate_orthogonal_points(
    theta1: List[torch.Tensor],
    theta2: List[torch.Tensor],
    pca_components: torch.Tensor,
    pca_mean: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Function to flatten a list of tensors into a 1D tensor
    def flatten_theta(theta: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.flatten() for t in theta])

    # Function to unflatten a 1D tensor into the structure of a given list of tensors
    def unflatten_theta(
        flat_tensor: torch.Tensor, structure: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        pointer = 0
        unflattened = []
        for t in structure:
            num_elements = t.numel()
            shape = t.shape
            unflattened_tensor = flat_tensor[pointer : pointer + num_elements].view(
                shape
            )
            unflattened.append(unflattened_tensor)
            pointer += num_elements
        return unflattened

    # Flatten the input thetas
    flat_theta1 = flatten_theta(theta1)
    flat_theta2 = flatten_theta(theta2)

    # Project to PCA space
    p1 = (flat_theta1 - pca_mean) @ pca_components.t()
    p2 = (flat_theta2 - pca_mean) @ pca_components.t()

    # Compute the vector between p2 and p1 in PCA space
    v = p2 - p1
    # Generate a perpendicular vector (rotated 90 degrees counterclockwise)
    u = torch.stack((-v[1], v[0]))
    # Calculate the midpoint between p1 and p2
    mid = (p1 + p2) / 2
    # Compute the other two points in PCA space
    p3 = mid + u / 2
    p4 = mid - u / 2

    # Inverse transform to original space
    flat_theta3 = p3 @ pca_components + pca_mean
    flat_theta4 = p4 @ pca_components + pca_mean

    # Unflatten back to the original structure
    theta3 = unflatten_theta(flat_theta3, theta1)
    theta4 = unflatten_theta(flat_theta4, theta1)

    return theta3, theta4


def find_closest_point(
    path1: List[List[torch.Tensor]], path2: List[List[torch.Tensor]]
) -> List[torch.Tensor]:
    min_dist = float("inf")
    closest_point = None

    for point1 in path1:
        flat_p1 = flatten_params(point1)
        for point2 in path2:
            flat_p2 = flatten_params(point2)
            dist = torch.norm(flat_p1 - flat_p2).item()
            if dist < min_dist:
                min_dist = dist
                closest_point = point1  # Prefer point from original path
    return closest_point


def closest_point_on_segment(
    P: torch.Tensor, A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """Find the closest point on segment AB to point P."""
    AP = P - A
    AB = B - A
    t = torch.dot(AP, AB) / torch.dot(AB, AB)
    t_clamped = torch.clamp(t, 0.0, 1.0)
    return A + t_clamped * AB


def calculate_curvature(path):
    flattened_path = [flatten_params(point) for point in path]
    if len(flattened_path) < 2:
        return 1.0

    piecewise_length = 0.0
    for i in range(len(flattened_path) - 1):
        diff = flattened_path[i + 1] - flattened_path[i]
        piecewise_length += torch.norm(
            diff, p=2
        ).item()

    start_point = flattened_path[0]
    end_point = flattened_path[-1]
    direct_length = torch.norm(end_point - start_point, p=2).item()

    if direct_length < 1e-10:
        return float("inf")

    curvature = piecewise_length / direct_length
    return curvature


def estimate_energy_gap_dss_pair(
    cls,
    loss_func,
    trainer,
    theta_a: List[torch.Tensor],
    theta_b: List[torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    threshold: float,
    max_depth: int,
    l1_lambda: float,
    max_epochs: int = 100,
    path_steps: int = 100,
    normalize_first_layer_weights: bool = False,
    **kwargs,
):
    kwargs.pop("normalize_first_layer_weights", None)
    barrier_losses = []
    hit_max_depth = []
    model = cls(**kwargs).to(device)
    path = dss(
        cls,
        model,
        loss_func,
        trainer,
        theta_a,
        theta_b,
        dataloader,
        threshold=threshold,
        max_depth=max_depth,
        losses=[],
        energy_gaps=None,
        barrier_losses=barrier_losses,
        hit_max_depth=hit_max_depth,
        l1_lambda=l1_lambda,
        max_epochs=max_epochs,
        path_steps=path_steps,
        normalize_first_layer_weights=normalize_first_layer_weights,
        **kwargs,
    )
    if hit_max_depth:
        min_barrier = min(barrier_losses) if barrier_losses else threshold
    else:
        min_barrier = threshold
    gap = max(0.0, min_barrier - threshold)
    return gap, min_barrier, path, bool(hit_max_depth), barrier_losses

def _cpuify(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpuify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpuify(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpuify(v) for v in obj)
    return obj

def _save_checkpoint(path, payload):
    if not path:
        return
    tmp_path = f"{path}.tmp"
    torch.save(_cpuify(payload), tmp_path)
    os.replace(tmp_path, path)

def _load_checkpoint(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Checkpoint load failed ({path}): {exc}")
        return None

def calculate_dss_statistics(
    cls,
    input_size,
    hidden_size,
    output_size,
    train_loader,
    loss_func,
    trainer,
    max_depth,
    l1_lambda,
    threshold=0.1,
    num=20,
    max_epochs=100,
    return_energy_gaps=False,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
):
    paths = []
    curvatures = []
    energy_gaps_all = [] if return_energy_gaps else None
    start_idx = 0
    if resume and checkpoint_path:
        state = _load_checkpoint(checkpoint_path)
        if state:
            paths = state.get("paths", [])
            curvatures = state.get("curvatures", [])
            if return_energy_gaps:
                energy_gaps_all = state.get("energy_gaps_all", [])
            start_idx = int(state.get("next_idx", len(curvatures)))
            start_idx = min(start_idx, num)
            if start_idx > 0:
                print(f"Resuming from checkpoint at {start_idx}/{num}")

    for i in range(start_idx, num):
        print(f"Iteration: {i}")
        torch.manual_seed(i*2)
        model = cls(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(device)

        trainer(
            model,
            train_loader,
            max_epochs=max_epochs,
            threshold=threshold,
            l1_lambda=l1_lambda,
        )
        theta1 = [p.detach().clone() for p in model.parameters()]
        print("start learned")

        torch.manual_seed(i*2 + 1)
        model = cls(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(device)
        trainer(
            model,
            train_loader,
            max_epochs=max_epochs,
            threshold=threshold,
            l1_lambda=l1_lambda,
        )
        theta2 = [p.detach().clone() for p in model.parameters()]
        print("end learned")

        losses = []
        energy_gaps = [] if return_energy_gaps else None
        path = dss(
            cls,
            model,
            loss_func,
            trainer,
            theta1,
            theta2,
            train_loader,
            threshold=threshold,
            max_depth=max_depth,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            losses=losses,
            energy_gaps=energy_gaps,
            l1_lambda=l1_lambda,
            max_epochs=max_epochs,
        )
        paths.append(path)
        curvatures.append(calculate_curvature(path))
        if return_energy_gaps:
            energy_gaps_all.append(max(energy_gaps) if energy_gaps else 0.0)
        print(f"Current curvature: {curvatures[-1]}")
        print(f"Current mean curvature: {np.array(curvatures).mean()}")
        if checkpoint_path and checkpoint_every > 0:
            is_due = ((i + 1) % checkpoint_every == 0) or (i + 1 == num)
            if is_due:
                _save_checkpoint(
                    checkpoint_path,
                    {
                        "kind": "calculate_dss_statistics",
                        "num": num,
                        "next_idx": i + 1,
                        "paths": paths,
                        "curvatures": curvatures,
                        "energy_gaps_all": energy_gaps_all,
                    },
                )
    if return_energy_gaps:
        return paths, curvatures, energy_gaps_all
    return paths, curvatures

def calculate_dss_statistics_abs(
    cls,
    input_size,
    hidden_size,
    output_size,
    train_loader,
    loss_func,
    trainer,
    max_depth,
    l1_lambda,
    num=20,
    max_epochs=100,
    return_energy_gaps=False,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
):
    paths = []
    curvatures = []
    energy_gaps_all = [] if return_energy_gaps else None
    start_idx = 0
    if resume and checkpoint_path:
        state = _load_checkpoint(checkpoint_path)
        if state:
            paths = state.get("paths", [])
            curvatures = state.get("curvatures", [])
            if return_energy_gaps:
                energy_gaps_all = state.get("energy_gaps_all", [])
            start_idx = int(state.get("next_idx", len(curvatures)))
            start_idx = min(start_idx, num)
            if start_idx > 0:
                print(f"Resuming from checkpoint at {start_idx}/{num}")

    for i in range(start_idx, num):
        print(f"Iteration: {i}")
        torch.manual_seed(i*2)
        model = cls(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(device)

        trainer(
            model,
            train_loader,
            max_epochs=max_epochs,
            threshold=float("-inf"),
            l1_lambda=l1_lambda,
        )
        theta1 = [p.detach().clone() for p in model.parameters()]
        start_loss = compute_avg_loss(model, train_loader, loss_func, l1_lambda)

        torch.manual_seed(i*2 + 1)
        model = cls(
            input_size=input_size, hidden_size=hidden_size, output_size=output_size
        ).to(device)
        trainer(
            model,
            train_loader,
            max_epochs=max_epochs,
            threshold=float("-inf"),
            l1_lambda=l1_lambda,
        )
        theta2 = [p.detach().clone() for p in model.parameters()]
        end_loss = compute_avg_loss(model, train_loader, loss_func, l1_lambda)

        losses = []
        energy_gaps = [] if return_energy_gaps else None
        path = dss(
            cls,
            model,
            loss_func,
            trainer,
            theta1,
            theta2,
            train_loader,
            threshold=max(start_loss, end_loss),
            max_depth=max_depth,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            losses=losses,
            energy_gaps=energy_gaps,
            l1_lambda=l1_lambda,
            max_epochs=max_epochs,
        )
        paths.append(path)
        curvatures.append(calculate_curvature(path))
        if return_energy_gaps:
            energy_gaps_all.append(max(energy_gaps) if energy_gaps else 0.0)
        print(f"Current curvature: {curvatures[-1]}")
        print(f"Current mean curvature: {np.array(curvatures).mean()}")
        if checkpoint_path and checkpoint_every > 0:
            is_due = ((i + 1) % checkpoint_every == 0) or (i + 1 == num)
            if is_due:
                _save_checkpoint(
                    checkpoint_path,
                    {
                        "kind": "calculate_dss_statistics_abs",
                        "num": num,
                        "next_idx": i + 1,
                        "paths": paths,
                        "curvatures": curvatures,
                        "energy_gaps_all": energy_gaps_all,
                    },
                )
    if return_energy_gaps:
        return paths, curvatures, energy_gaps_all
    return paths, curvatures

def train_and_track_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    init_params: List[torch.Tensor],
    lr: float = 0.001,
    max_epochs: int = 100,
    l1_lambda=0.01,
    normalize_first_layer_weights: bool = False,
) -> List[float]:
    model = model.to(device)
    with torch.no_grad():
        for param, p in zip(model.parameters(), init_params):
            param.copy_(p.to(param.device))

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    l1_params = get_l1_params(model)

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            l1_reg = sum(torch.sum(torch.abs(p)) for p in l1_params)
            loss += l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            if normalize_first_layer_weights:
                normalize_first_layer_fast(model)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

    return losses


def train_and_track_loss_classifier(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    init_params: List[torch.Tensor],
    lr: float = 0.001,
    max_epochs: int = 100,
    l1_lambda=0.01,
    normalize_first_layer_weights: bool = False,
) -> List[float]:
    model = model.to(device)
    with torch.no_grad():
        for param, p in zip(model.parameters(), init_params):
            param.copy_(p.to(param.device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    l1_params = get_l1_params(model)

    for epoch in range(max_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            l1_reg = sum(torch.sum(torch.abs(p)) for p in l1_params)
            loss += l1_lambda * l1_reg
            loss.backward()
            optimizer.step()
            if normalize_first_layer_weights:
                normalize_first_layer_fast(model)

            total_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            correct += (predicted == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

    return losses


def set_params(model, theta):
    with torch.no_grad():
        for p, v in zip(model.parameters(), theta):
            p.copy_(v.to(p.device))

def get_init_params(cls, seed, **model_kwargs):
    torch.manual_seed(seed)
    model = cls(**model_kwargs).to(device)
    theta = [p.detach().clone() for p in model.parameters()]
    return theta

def first_epoch_below(losses, threshold):
    for i, v in enumerate(losses):
        if v <= threshold:
            return i
    return None

def paired_test(x, y, alpha=0.05, alternative='less', use_shapiro=True):
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    cohen_dz = mean_diff / sd if (len(diff) > 1 and sd > 0 and not np.isnan(sd)) else np.nan

    use_t = True
    if use_shapiro and len(diff) >= 3:
        try:
            use_t = stats.shapiro(diff).pvalue > alpha
        except Exception:
            pass

    if use_t and len(diff) >= 2:
        t_stat, p_two = stats.ttest_rel(x, y)
        if alternative == 'less':
            p = p_two/2 if mean_diff < 0 else 1 - p_two/2
        elif alternative == 'greater':
            p = p_two/2 if mean_diff > 0 else 1 - p_two/2
        else:
            p = p_two
        test_name = "Paired t-test"
    else:
        alt = alternative if alternative in ('less','greater','two-sided') else 'two-sided'
        stat, p = stats.wilcoxon(x, y, alternative=alt, zero_method='wilcox')
        test_name = "Wilcoxon signed-rank"

    rng = np.random.default_rng(0)
    if len(diff) > 0:
        bs = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        ci_low, ci_high = [float(v) for v in np.percentile(bs, [2.5, 97.5])]
    else:
        ci_low = ci_high = np.nan

    return test_name, float(p), float(cohen_dz), mean_diff, ci_low, ci_high

def report_pair(name, x, y, alpha=0.05, alternative='less'):
    test_name, p, d, mdiff, ci_l, ci_h = paired_test(x, y, alpha=alpha, alternative=alternative)
    print(f"\n{name}: {test_name} (H1: opt {alternative} base)")
    print(f"mean(opt - base) = {mdiff:.6f}  [95% CI: {ci_l:.6f}, {ci_h:.6f}]")
    print(f"Cohen's dz = {d:.3f}")
    print(f"p-value = {p:.6g}")
    if p < alpha:
        print("⇒ The difference is statistically significant")
    else:
        print("⇒ No statistically significant difference")

def build_optimal_point_pipeline(
    cls, loss_func, trainer_to_thr, train_loader,
    threshold, l1_lambda, max_depth,
    input_size, hidden_size, output_size,
    max_epochs_dss=150,
    base_seed=1000
):
    t0 = time.perf_counter()

    torch.manual_seed(base_seed + 1)
    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    start = [p.detach().clone() for p in model.parameters()]
    trainer_to_thr(model, train_loader, threshold=threshold, l1_lambda=l1_lambda)
    theta1 = [p.detach().clone() for p in model.parameters()]

    torch.manual_seed(base_seed + 2)
    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    end = [p.detach().clone() for p in model.parameters()]
    trainer_to_thr(model, train_loader, threshold=threshold, l1_lambda=l1_lambda)
    theta2 = [p.detach().clone() for p in model.parameters()]

    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    path3 = dss(
        cls, model, loss_func, trainer_to_thr,
        theta1, theta2,
        train_loader,
        threshold=threshold, max_depth=max_depth,
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        losses=[], l1_lambda=l1_lambda
    )
    path3 = [start] + path3 + [end]

    params_list = [flatten_params(theta).cpu().numpy() for theta in path3]
    pca = PCA(n_components=2)
    pca.fit(params_list)
    pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)

    start2, end2 = generate_orthogonal_points(path3[1], path3[-2], pca_components, pca_mean)

    torch.manual_seed(base_seed + 3)
    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    set_params(model, start2)
    trainer_to_thr(model, train_loader, threshold=threshold, l1_lambda=l1_lambda)
    theta3 = [p.detach().clone() for p in model.parameters()]

    torch.manual_seed(base_seed + 4)
    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    set_params(model, end2)
    trainer_to_thr(model, train_loader, threshold=threshold, l1_lambda=l1_lambda)
    theta4 = [p.detach().clone() for p in model.parameters()]

    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    path_perp2 = dss(
        cls, model, loss_func, trainer_to_thr,
        theta3, theta4,
        train_loader,
        threshold,
        max_depth=max_depth,
        input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        losses=[], l1_lambda=l1_lambda, max_epochs=max_epochs_dss
    )
    path_perp2 = [start2] + path_perp2 + [end2]

    optimal_point = find_closest_point(path3, path_perp2)

    t_search = time.perf_counter() - t0
    return optimal_point, float(t_search), path3, path_perp2

def train_fixed_epochs_with_timing(model, train_loader, init_theta, epochs, trainer_with_tracking):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    losses = trainer_with_tracking(model, train_loader, init_theta, max_epochs=epochs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return losses, float(elapsed)

def pilot_epochs_per_second(cls, seed, train_loader, epochs_pilot, trainer_with_tracking,
                            input_size, hidden_size, output_size):
    theta_base = get_init_params(cls, seed=seed, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    _ = trainer_with_tracking(model, train_loader, theta_base, max_epochs=epochs_pilot)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    t_per_epoch = elapsed / max(1, epochs_pilot)
    return float(t_per_epoch)

@dataclass
class TrialMetrics:
    trial: int
    t_opt_search: float
    t_opt_train_fixed: float
    t_base_train_budget: float
    t_to_thr_opt: float
    t_to_thr_base: float
    opt_fixed_epochs: int
    base_epochs_for_budget: int
    final_loss_opt_fixed: float
    final_loss_base_budget: float
    min_loss_opt_fixed: float
    min_loss_base_budget: float
    epochs_to_thr_opt_in_fixed: float  
    epochs_to_thr_base_in_budget: float  
    budget_kind: str  

def run_one_trial(
    trial_idx,
    cls,
    loss_func,
    trainer_to_thr,               
    trainer_with_tracking,        
    train_loader,
    threshold,
    l1_lambda,
    max_depth,
    input_size, hidden_size, output_size,
    opt_fixed_epochs=300,
    budget_mode='search_plus_train',  # 'train_only' | 'search_plus_train'
    pilot_epochs=10,
):
    base_seed = 10000 + trial_idx * 1000

    optimal_point, t_opt_search, _, _ = build_optimal_point_pipeline(
        cls, loss_func, trainer_to_thr, train_loader,
        threshold, l1_lambda, max_depth,
        input_size, hidden_size, output_size,
        max_epochs_dss=150,
        base_seed=base_seed
    )

    torch.manual_seed(base_seed + 10)
    model_opt = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    opt_losses, t_opt_train_fixed = train_fixed_epochs_with_timing(
        model_opt, train_loader, optimal_point, opt_fixed_epochs, trainer_with_tracking
    )
    final_loss_opt_fixed = float(opt_losses[-1])
    min_loss_opt_fixed = float(np.min(opt_losses))
    ep_thr_opt_in_fixed = first_epoch_below(opt_losses, threshold)
    ep_thr_opt_in_fixed = float(ep_thr_opt_in_fixed) if ep_thr_opt_in_fixed is not None else float('nan')

    if budget_mode == 'train_only':
        budget_seconds = t_opt_train_fixed
    elif budget_mode == 'search_plus_train':
        budget_seconds = t_opt_search + t_opt_train_fixed
    else:
        raise ValueError("budget_mode must be 'train_only' or 'search_plus_train'.")

    t_per_epoch_base = pilot_epochs_per_second(
        cls, seed=base_seed + 20, train_loader=train_loader, epochs_pilot=pilot_epochs,
        trainer_with_tracking=trainer_with_tracking,
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )
    base_epochs_for_budget = max(1, int(math.ceil(budget_seconds / max(t_per_epoch_base, 1e-9))))

    theta_base_final = get_init_params(cls, seed=base_seed + 21, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model_base = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    base_losses_budget, t_base_train_budget = train_fixed_epochs_with_timing(
        model_base, train_loader, theta_base_final, base_epochs_for_budget, trainer_with_tracking
    )
    final_loss_base_budget = float(base_losses_budget[-1])
    min_loss_base_budget = float(np.min(base_losses_budget))
    ep_thr_base_in_budget = first_epoch_below(base_losses_budget, threshold)
    ep_thr_base_in_budget = float(ep_thr_base_in_budget) if ep_thr_base_in_budget is not None else float('nan')

    torch.manual_seed(base_seed + 30)
    model_opt_thr = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    set_params(model_opt_thr, optimal_point)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    trainer_to_thr(model_opt_thr, train_loader, threshold=threshold, l1_lambda=l1_lambda)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_to_thr_opt = float(time.perf_counter() - t0)

    # BASE
    theta_base_thr = get_init_params(cls, seed=base_seed + 31, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model_base_thr = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    trainer_to_thr(model_base_thr, train_loader, threshold=threshold, l1_lambda=l1_lambda)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_to_thr_base = float(time.perf_counter() - t1)

    tm = TrialMetrics(
        trial=trial_idx,
        t_opt_search=float(t_opt_search),
        t_opt_train_fixed=float(t_opt_train_fixed),
        t_base_train_budget=float(t_base_train_budget),
        t_to_thr_opt=t_to_thr_opt,
        t_to_thr_base=t_to_thr_base,
        opt_fixed_epochs=int(opt_fixed_epochs),
        base_epochs_for_budget=int(base_epochs_for_budget),
        final_loss_opt_fixed=final_loss_opt_fixed,
        final_loss_base_budget=final_loss_base_budget,
        min_loss_opt_fixed=min_loss_opt_fixed,
        min_loss_base_budget=min_loss_base_budget,
        epochs_to_thr_opt_in_fixed=ep_thr_opt_in_fixed,
        epochs_to_thr_base_in_budget=ep_thr_base_in_budget,
        budget_kind=budget_mode,
    )
    return tm


def run_experiment_and_test(
    num_runs,
    cls,
    loss_func,
    trainer_to_thr,               # train_model_to_threshold_classifier
    trainer_with_tracking,        # train_and_track_loss_classifier
    train_loader,
    threshold,
    l1_lambda,
    max_depth,
    input_size, hidden_size, output_size,
    opt_fixed_epochs=300,
    budget_mode='search_plus_train',  # 'train_only' | 'search_plus_train'
    pilot_epochs=10,
    alpha=0.05,
    save_prefix="opt_vs_base_equal_wallclock",
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
    resume: bool = True,
    save_partial_csv: bool = True,
):
    if checkpoint_path is None:
        checkpoint_path = f"{save_prefix}_{budget_mode}_checkpoint.pt"

    all_metrics = []
    start_idx = 0
    if resume and checkpoint_path:
        state = _load_checkpoint(checkpoint_path)
        if state:
            all_metrics = state.get("metrics", [])
            start_idx = min(len(all_metrics), num_runs)
            if start_idx > 0:
                print(f"Resuming from checkpoint at {start_idx}/{num_runs}")

    for r in range(start_idx, num_runs):
        print(f"=== Trial {r+1}/{num_runs} ===")
        tm = run_one_trial(
            r, cls, loss_func, trainer_to_thr, trainer_with_tracking, train_loader,
            threshold, l1_lambda, max_depth, input_size, hidden_size, output_size,
            opt_fixed_epochs=opt_fixed_epochs, budget_mode=budget_mode, pilot_epochs=pilot_epochs
        )
        all_metrics.append(asdict(tm))
        if checkpoint_path and checkpoint_every > 0:
            is_due = ((r + 1) % checkpoint_every == 0) or (r + 1 == num_runs)
            if is_due:
                _save_checkpoint(
                    checkpoint_path,
                    {
                        "kind": "run_experiment_and_test",
                        "num_runs": num_runs,
                        "next_idx": r + 1,
                        "metrics": all_metrics,
                        "budget_mode": budget_mode,
                        "save_prefix": save_prefix,
                    },
                )
        if save_partial_csv:
            csv_path = f"{save_prefix}_{budget_mode}_raw.csv"
            pd.DataFrame(all_metrics).to_csv(csv_path, index=False)

    df = pd.DataFrame(all_metrics)
    csv_path = f"{save_prefix}_{budget_mode}_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nOld metrics saved to {csv_path}")

    report_pair("FINAL loss with equal wall-clock budget",
                df["final_loss_opt_fixed"].values, df["final_loss_base_budget"].values,
                alpha=alpha, alternative='less')

    report_pair("MINIMUM loss with equal wall-clock budget",
                df["min_loss_opt_fixed"].values, df["min_loss_base_budget"].values,
                alpha=alpha, alternative='less')

    report_pair("Time to threshold (early stop)",
                df["t_to_thr_opt"].values, df["t_to_thr_base"].values,
                alpha=alpha, alternative='less')

    mask_thr = (~df["epochs_to_thr_opt_in_fixed"].isna()) & (~df["epochs_to_thr_base_in_budget"].isna())
    if mask_thr.sum() >= 2:
        report_pair("Epochs to threshold (on fixed opt and budget base)",
                    df.loc[mask_thr, "epochs_to_thr_opt_in_fixed"].values,
                    df.loc[mask_thr, "epochs_to_thr_base_in_budget"].values,
                    alpha=alpha, alternative='less')
    else:
        print("\nEpos to threshold: not enough observations for pairwise comparison.")

    def med_ratio(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.median(a / b))
    print("\nMedian ratios (opt/base) with equal wall-clock:")
    print(f"  Final loss: {med_ratio(df['final_loss_opt_fixed'].values, df['final_loss_base_budget'].values):.3f}×")
    print(f"  Min loss:   {med_ratio(df['min_loss_opt_fixed'].values, df['min_loss_base_budget'].values):.3f}×")
    print(f"  Train time to thr: {med_ratio(df['t_to_thr_opt'].values, df['t_to_thr_base'].values):.3f}×")

    return df
