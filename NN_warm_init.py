# GPU-accelerated neural net with warm-start capability

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple, List

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
    def __init__(self, input_dim: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden

        layers = []
        prev_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                if i < len(self.net) - 1:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def reset_last_layer(self):
        last_linear = None
        for m in reversed(list(self.net)):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.xavier_uniform_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)


class LogisticSumLoss(nn.Module):
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
    """GPU-accelerated logistic NN with warm-start capability."""

    def __init__(
            self,
            input_dim: int,
            hidden: Tuple[int, ...] = (64, 64),
            l2: float = 1e-4,
            seed: int = 42,
            init_from: Optional['NumpyLogisticNN'] = None,
            init_WB: Optional[Tuple[List, List]] = None,
            reset_last_layer: bool = False
    ):
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

        if init_from is not None:
            self._copy_params_from(init_from)
        if init_WB is not None:
            self._set_params_from_wb(init_WB)
        if reset_last_layer:
            self.model.reset_last_layer()

    def _copy_params_from(self, other: 'NumpyLogisticNN'):
        with torch.no_grad():
            for p_self, p_other in zip(self.model.parameters(), other.model.parameters()):
                p_self.copy_(p_other)

    def _set_params_from_wb(self, init_WB: Tuple[List, List]):
        W, B = init_WB
        with torch.no_grad():
            param_list = list(self.model.parameters())
            idx = 0
            for Wi, Bi in zip(W, B):
                param_list[idx].copy_(to_tensor(Wi.T, self.device))  # transpose for PyTorch
                param_list[idx + 1].copy_(to_tensor(Bi, self.device))
                idx += 2

    def get_params(self, copy: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        W, B = [], []
        for name, p in self.model.named_parameters():
            arr = to_numpy(p)
            if 'weight' in name:
                W.append(arr.T.copy() if copy else arr.T)
            elif 'bias' in name:
                B.append(arr.copy() if copy else arr)
        return W, B

    def set_params(self, W: List, B: List, reset_last_layer: bool = False):
        self._set_params_from_wb((W, B))
        if reset_last_layer:
            self.model.reset_last_layer()

    def fit(
            self,
            X1,
            X2,
            epochs: int = 200,
            batch_size: int = 128,
            lr: float = 1e-3,
            verbose: int = 0,
            shuffle: bool = True,
            X_test: Optional[np.ndarray] = None,
            es_rel: float = 0.01,
            es_min_epochs: int = 1,
            es_norm_ord: int = 2,
            es_abs_floor: float = 1e-12
    ):
        X1 = np.asarray(X1, dtype=np.float32)
        X2 = np.asarray(X2, dtype=np.float32)
        X = np.vstack([X1, X2])
        y = np.concatenate([np.ones(len(X1), dtype=np.float32),
                            -np.ones(len(X2), dtype=np.float32)])

        X_t = to_tensor(X, self.device)
        y_t = to_tensor(y, self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.l2)

        # Early stopping setup
        if X_test is not None:
            Xtest = np.asarray(X_test, dtype=np.float32)
            if Xtest.ndim == 1:
                Xtest = Xtest[None, :]
            Xtest_t = to_tensor(Xtest, self.device)
            self.model.eval()
            with torch.no_grad():
                f_test_prev = to_numpy(self.model(Xtest_t))
            prev_norm = max(float(np.linalg.norm(f_test_prev, ord=es_norm_ord)), es_abs_floor)
        else:
            Xtest_t = None
            f_test_prev = None
            prev_norm = None

        self.model.train()
        for ep in range(1, epochs + 1):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                f = self.model(xb)
                loss = self.criterion(f, yb)
                loss.backward()
                self.optimizer.step()

            # Early stopping check
            if Xtest_t is not None and ep >= es_min_epochs:
                self.model.eval()
                with torch.no_grad():
                    f_test_curr = to_numpy(self.model(Xtest_t))
                self.model.train()

                diff_norm = float(np.linalg.norm(f_test_curr - f_test_prev, ord=es_norm_ord))
                threshold = es_rel * prev_norm
                if diff_norm < threshold:
                    if verbose:
                        print(f"Early stop at epoch {ep}: ||Δf_test||={diff_norm:.6e}")
                    return self
                f_test_prev = f_test_curr
                prev_norm = max(float(np.linalg.norm(f_test_prev, ord=es_norm_ord)), es_abs_floor)

        return self

    def f_theta(self, x):
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
