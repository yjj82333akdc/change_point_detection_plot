# GPU-accelerated neural net minimizing:
#   sum_{X1} log(1+exp(-f(x))) + sum_{X2} log(1+exp(f(x)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE


def to_tensor(x, device=None):
    device = device or DEVICE
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


class NNModel(nn.Module):
    """MLP with ReLU hidden layers and linear output (f_theta)."""

    def __init__(self, input_dim: int, hidden: tuple = (64, 64)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # scalar output
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LogisticSumLoss(nn.Module):
    """Loss: sum softplus(-y * f) where y in {+1, -1}."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, f, y):
        loss = torch.nn.functional.softplus(-y * f)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class NumpyLogisticNN:
    """GPU-accelerated logistic NN trainer with NumPy-compatible interface."""

    def __init__(self, input_dim: int, hidden: tuple = (64, 64), l2: float = 1e-4, seed: int = 42):
        self.input_dim = int(input_dim)
        self.hidden = tuple(hidden)
        self.l2 = float(l2)
        self.seed = seed
        self.device = get_device()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.model = NNModel(input_dim, hidden).to(self.device)
        self.optimizer = None
        self.criterion = LogisticSumLoss(reduction='mean')

    def fit(self, X1, X2, epochs: int = 200, batch_size: int = 128, lr: float = 1e-3, verbose: int = 1):
        X1 = np.asarray(X1, dtype=np.float32)
        X2 = np.asarray(X2, dtype=np.float32)
        X = np.vstack([X1, X2])
        y = np.concatenate([np.ones(len(X1), dtype=np.float32),
                            -np.ones(len(X2), dtype=np.float32)])

        X_t = to_tensor(X, self.device)
        y_t = to_tensor(y, self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.l2)

        self.model.train()
        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                f = self.model(xb)
                loss = self.criterion(f, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(xb)

            if verbose and (ep % max(1, epochs // 20) == 0 or ep == 1):
                self.model.eval()
                with torch.no_grad():
                    f_full = self.model(X_t)
                    acc = ((f_full > 0) == (y_t > 0)).float().mean().item()
                self.model.train()
                print(f"Epoch {ep:4d} | loss={epoch_loss / len(X):.3f} | acc={acc:.3f}")

        return self

    def f_theta(self, x):
        """Evaluate f_theta(x). Returns float for single input, array for batch."""
        x = np.asarray(x, dtype=np.float32)
        single = (x.ndim == 1)
        if single:
            x = x[None, :]

        self.model.eval()
        with torch.no_grad():
            x_t = to_tensor(x, self.device)
            f = self.model(x_t)

        result = to_numpy(f)
        return float(result[0]) if single else result

    def get_params(self, copy: bool = True):
        """Get model parameters as list of numpy arrays."""
        params = []
        for p in self.model.parameters():
            arr = to_numpy(p)
            params.append(arr.copy() if copy else arr)
        return params

    def set_params(self, params):
        """Set model parameters from list of numpy arrays."""
        with torch.no_grad():
            for p, arr in zip(self.model.parameters(), params):
                p.copy_(to_tensor(arr, self.device))
