"""Refactored forward model module.

Provides `ForwardMLP`, `generate_curve`, dataset utilities and a train entrypoint.
This module is import-safe (won't run training on import).
"""
import numpy as np
from typing import Tuple

# Lazy imports for optional heavy dependencies (torch, sklearn). This lets
# lightweight utilities like `make_dataset` be imported without installing
# those packages, which is useful for quick smoke tests.
_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _torch_available = True
except Exception:
    nn = None
    optim = None


if _torch_available:
    class ForwardMLP(nn.Module):
        def __init__(self, input_dim=2, output_dim=20, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)
else:
    class ForwardMLP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ForwardMLP requires torch; install torch to use training utilities")


def generate_curve(N_sutures, rate_class, T=20, d0=10.0, df=20.0, noise_level=0.01):
    rate_multiplier = {0: 0.8, 1: 1.0, 2: 1.2}[rate_class]
    t = np.linspace(0, 1, T)
    suture_fraction = 1.0 / N_sutures
    curve = np.zeros(T)
    for i in range(N_sutures):
        t_start = i / N_sutures
        t_end = (i + 1) / N_sutures
        s = 1 / (1 + np.exp(-10 * (t - t_start) * rate_multiplier))
        s_next = 1 / (1 + np.exp(-10 * (t - t_end) * rate_multiplier))
        curve += suture_fraction * (s - s_next)
    curve = d0 + (df - d0) * np.cumsum(curve)
    curve = np.clip(curve, d0, df)
    curve += np.random.normal(0, noise_level, size=T)
    curve_smooth = np.convolve(curve, np.ones(3) / 3, mode="same")
    curve_smooth = np.maximum.accumulate(curve_smooth)
    return curve_smooth


def make_dataset(N_samples=500, T=20, random_state=42):
    rng = np.random.RandomState(random_state)
    X_list, Y_list = [], []
    for _ in range(N_samples):
        n = rng.randint(1, 10)
        r = rng.randint(0, 3)
        curve = generate_curve(n, r, T=T)
        X_list.append([n, r])
        Y_list.append(curve)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y


def get_dataloaders(batch_size=32, test_size=0.2, val_ratio=0.1, **kwargs):
    """Return PyTorch dataloaders (lazy-imports torch and sklearn).

    Raises a helpful error if required packages are not installed.
    """
    if not _torch_available:
        raise RuntimeError("get_dataloaders requires torch; install torch to use dataloaders")

    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        raise RuntimeError("get_dataloaders requires scikit-learn; install scikit-learn or call make_dataset directly")

    X, Y = make_dataset(**kwargs)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=test_size, random_state=42)
    val_fraction_of_temp = val_ratio / (1.0 - test_size)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def physics_loss(pred_curve, true_curve=None, alpha=1.0, beta=0.1, gamma=0.1):
    if not _torch_available:
        raise RuntimeError("physics_loss requires torch; install torch to use training utilities")

    device = pred_curve.device
    data_loss = torch.tensor(0.0, device=device)
    final_penalty = torch.tensor(0.0, device=device)
    if true_curve is not None:
        data_loss = torch.mean((pred_curve - true_curve) ** 2)
        final_penalty = torch.mean((pred_curve[:, -1] - true_curve[:, -1]) ** 2)
    diffs = pred_curve[:, 1:] - pred_curve[:, :-1]
    monotonic_penalty = torch.mean(torch.relu(-diffs))
    second_diffs = pred_curve[:, 2:] - 2 * pred_curve[:, 1:-1] + pred_curve[:, :-2]
    smoothness_penalty = torch.mean(second_diffs ** 2)
    total_loss = alpha * data_loss + beta * (monotonic_penalty + smoothness_penalty) + gamma * final_penalty
    return total_loss


def train(model: "nn.Module" = None, epochs: int = 50, device: str = "cpu", **dataset_kwargs):
    if not _torch_available:
        raise RuntimeError("train requires torch; install torch to use training utilities")

    if model is None:
        model = ForwardMLP()
    model.to(device)
    train_loader, val_loader, _ = get_dataloaders(**dataset_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            pred = model(xb)
            loss = physics_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                xb, yb = next(iter(val_loader))
                val_pred = model(xb.to(device).float())
                val_loss = physics_loss(val_pred, yb.to(device).float())
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss.item():.4f}")
    return model


if __name__ == "__main__":
    trained = train(epochs=50, batch_size=32)
