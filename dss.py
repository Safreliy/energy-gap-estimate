import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import random
import numpy as np
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
) -> Tuple[float, float, List[torch.Tensor]]:
    max_loss, t_max = -float("inf"), 0.0
    theta_at_max = None

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
                theta_at_max = [
                    interp_params[name].detach().clone() for name in param_names
                ]

    if theta_at_max is None:
        theta_at_max = [p.detach().clone() for p in theta_a]

    return max_loss, t_max, theta_at_max


def max_loss_along_piecewise_path(
    model: nn.Module,
    loss_func,
    path: List[List[torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    steps: int = 100,
    l1_lambda: float = 0.01,
) -> float:
    """Evaluate the maximum objective over every segment of a returned path."""
    if len(path) < 2:
        raise ValueError("A piecewise-linear path needs at least two nodes.")
    maxima = [
        max_loss_along_path(
            model,
            loss_func,
            theta_a,
            theta_b,
            dataloader,
            steps=steps,
            l1_lambda=l1_lambda,
        )[0]
        for theta_a, theta_b in zip(path[:-1], path[1:])
    ]
    return max(maxima)

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
    loss, t, theta_peak = max_loss_along_path(
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
    theta_mid = [p.detach().clone() for p in theta_peak]
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
    path_barrier = max_loss_along_piecewise_path(
        model,
        loss_func,
        path,
        dataloader,
        steps=path_steps,
        l1_lambda=l1_lambda,
    )
    gap = max(0.0, path_barrier - threshold)
    return gap, path_barrier, path, bool(hit_max_depth), barrier_losses
