"""pdf_baselines_paper_fast.py

Paper-faithful baselines (as close as practical in this experiment setting)

Includes two detectors:

(1) Scan B-statistic (kernel MMD) style online detector
    Reference: "Scan B-statistics for kernel change-point detection" (arXiv:1507.01279)
    - Uses an averaged MMD^2_u over N reference blocks (Eq. 3.3)
    - Uses an empirical variance standardization Z / sqrt(Var(Z)) (paper uses Var(Z_{B0,t}))
      Here we approximate Var by the sample variance across the N blockwise MMD values.

(2) NN-CUSUM
    Reference: "NN-CUSUM" (arXiv:2210.17312)
    - Uses an NN-based increment eta_t (difference of mean scores on test splits)
    - Uses CUSUM recursion S_t = max(S_{t-1} + (eta_t - D), 0)

Threshold calibration (alpha-level) is intentionally performed in the *main experiment script*
by Monte Carlo under the null:
    b := Quantile_{1-alpha}( max_{t in horizon} statistic_t )
This matches how both papers discuss controlling Type-I error (or false alarm probability) via
either theory or simulation.

This module provides:
- detector implementations
- helper utilities to compute max-statistic over a time horizon
- helper utility to compute max-CUSUM from a precomputed eta-series
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List


# ----------------------------
# Paper-style ARL threshold for online ScanB
# ----------------------------
def _std_norm_pdf(x: float) -> float:
    return float(math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))

def _std_norm_cdf(x: float) -> float:
    # Φ(x) via erf
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))

def nu_approx(x: float) -> float:
    """Approximation of Siegmund's ν(x) used in scan-statistic tail/ARL formulas.

    We use the simple approximation (Siegmund & Yakir 2007; see e.g. Kuriki et al. 2014 Remark 1):
        ν(x) ≈ (2/x) (Φ(x/2) - 1/2) / ( (x/2) Φ(x/2) - φ(x/2) ),  x>0
    and ν(0)=1.

    Notes:
    - ν(x) ∈ (0,1] and increases to 1 as x→∞.
    - We clip numerically to avoid division-by-zero for small x.
    """
    if x <= 0.0:
        return 1.0
    u = 0.5 * x
    Phi = _std_norm_cdf(u)
    phi = _std_norm_pdf(u)
    num = (2.0 / x) * (Phi - 0.5)
    den = (u * Phi - phi)
    if den <= 1e-12:
        return 1.0
    v = num / den
    # numerical safety: ν(x) ≤ 1
    if not math.isfinite(v):
        return 1.0
    return float(min(max(v, 1e-6), 1.0))

def scanb_arl_theory(b: float, B0: int) -> float:
    """Theorem 4.2 ARL approximation for the online standardized ScanB statistic.

    From arXiv:1507.01279 (v5), Thm 4.2:
        E[T] ≈ exp(b^2/2)/b * [ (2B0-1) / sqrt(2π B0(B0-1)) * ν( b * sqrt( 2(2B0-1)/(B0(B0-1)) ) ) ]^{-1}
    valid as b→∞.

    We use the ν approximation above.
    """
    B0 = int(B0)
    if B0 < 2:
        raise ValueError("B0 must be >=2")
    b = float(b)
    if b <= 0:
        return 0.0
    c = (2.0 * B0 - 1.0) / math.sqrt(2.0 * math.pi * B0 * (B0 - 1.0))
    arg = b * math.sqrt(2.0 * (2.0 * B0 - 1.0) / (B0 * (B0 - 1.0)))
    return float(math.exp(0.5 * b * b) / b / (c * nu_approx(arg)))

def scanb_b_from_arl(target_arl: float, B0: int, *, max_iter: int = 80) -> float:
    """Solve for b such that scanb_arl_theory(b,B0) ~= target_arl via bisection."""
    target_arl = float(target_arl)
    if target_arl <= 1.0:
        raise ValueError("target_arl must be > 1")
    B0 = int(B0)

    lo, hi = 1.0, 10.0
    # expand hi until ARL(hi) >= target
    while scanb_arl_theory(hi, B0) < target_arl and hi < 50.0:
        hi *= 1.25
    if scanb_arl_theory(hi, B0) < target_arl:
        # extremely large target ARL, just return hi (best effort)
        return float(hi)

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        arl_mid = scanb_arl_theory(mid, B0)
        if arl_mid < target_arl:
            lo = mid
        else:
            hi = mid
    return float(hi)

import numpy as np
import math

# -------------------------
# Kernel MMD utilities
# -------------------------

def _sqdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(X2 + Y2 - 2.0 * (X @ Y.T), 0.0)

def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    d2 = _sqdist(X, Y)
    return np.exp(-d2 / (2.0 * (sigma ** 2))).astype(np.float32)

def median_heuristic_sigma(X: np.ndarray, max_points: int = 200, seed: int = 0) -> float:
    """Median pairwise distance heuristic (subsampled)."""
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    rng = np.random.default_rng(int(seed))
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    # pairwise distances
    d2 = _sqdist(Xs, Xs)
    iu = np.triu_indices(d2.shape[0], k=1)
    vals = np.sqrt(d2[iu] + 1e-12)
    med = float(np.median(vals)) if vals.size else 1.0
    return max(med, 1e-3)

def mmd_u2_rbf(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2_u with RBF kernel."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    m = X.shape[0]
    n = Y.shape[0]
    if m < 2 or n < 2:
        return 0.0

    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    # remove diagonal
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = np.sum(Kxx) / (m * (m - 1.0))
    term_yy = np.sum(Kyy) / (n * (n - 1.0))
    term_xy = 2.0 * np.sum(Kxy) / (m * n)
    return float(term_xx + term_yy - term_xy)

# -------------------------
# ScanB (paper-faithful-ish)
# -------------------------

@dataclass
class ScanBConfig:
    B0: int = 50          # block size
    N_blocks: int = 8     # number of reference blocks to average
    t_step: int = 1       # stride for scanning
    sigma: Optional[float] = None  # if None, median heuristic on training data

class ScanBDetector:
    """Scan-B style detector using averaged MMD^2_u across reference blocks.

    Paper alignment:
    - Z_t = (1/N) sum_i MMD^2_u(X_i, Y_t)   (Eq. 3.3)
    - Standardization: Z_t / sqrt(Var(Z_t)) (paper uses Var(Z_{B0,t}))
      Here: Var(Z_t) ≈ Var_i(MMD_i)/N using the sample variance across reference blocks.
    """

    def __init__(self, cfg: Optional[ScanBConfig] = None, seed: int = 0):
        self.cfg = cfg if cfg is not None else ScanBConfig()
        self.seed = int(seed)
        self.sigma: Optional[float] = None
        self._ref_blocks: Optional[np.ndarray] = None  # (N_blocks, B0, dim)

    def fit_reference(self, X0: np.ndarray, sigma_fit_max: int = 200) -> None:
        X0 = np.asarray(X0, dtype=np.float32)
        if X0.ndim != 2:
            raise ValueError("X0 must be 2D (n,dim)")
        n0, dim = X0.shape
        B0 = int(self.cfg.B0)
        N = int(self.cfg.N_blocks)
        need = N * B0
        if n0 < need:
            raise ValueError(f"Need at least N_blocks*B0={need} reference samples, got {n0}")

        rng = np.random.default_rng(self.seed + 123)

        # paper: sample without replacement from the reference pool (when possible)
        idx = rng.choice(n0, size=need, replace=False) if n0 >= need else rng.integers(0, n0, size=need)
        Xsel = X0[idx]
        self._ref_blocks = Xsel.reshape(N, B0, dim)

        # sigma
        if self.cfg.sigma is None:
            self.sigma = float(median_heuristic_sigma(X0, max_points=int(sigma_fit_max), seed=self.seed + 7))
        else:
            self.sigma = float(self.cfg.sigma)

    def Z_std(self, Y_block: np.ndarray) -> float:
        """Standardized scan-B statistic at a time t for a given block Y."""
        if self._ref_blocks is None:
            raise RuntimeError("Call fit_reference(X0) first")
        Y_block = np.asarray(Y_block, dtype=np.float32)
        vals = np.array([mmd_u2_rbf(self._ref_blocks[i], Y_block, float(self.sigma)) for i in range(self._ref_blocks.shape[0])],
                        dtype=np.float32)
        mean = float(np.mean(vals))
        # Var(mean) ≈ Var(vals)/N
        N = max(int(vals.size), 1)
        var_blocks = float(np.var(vals, ddof=1)) if vals.size >= 2 else 0.0
        var_mean = var_blocks / max(N, 1)
        denom = math.sqrt(var_mean + 1e-12)
        return float(mean / denom) if denom > 0 else 0.0

    def max_statistic(self, X: np.ndarray, t_start: int, t_end: int) -> float:
        """Return max_{t in [t_start,t_end]} Z_std(t), where Z_std uses the last B0 points as block."""
        if self._ref_blocks is None:
            raise RuntimeError("Call fit_reference(X0) first")
        X = np.asarray(X, dtype=np.float32)
        T = X.shape[0]
        t_end = min(int(t_end), T)
        B0 = int(self.cfg.B0)

        maxZ = -np.inf
        for t in range(int(t_start), t_end + 1, int(self.cfg.t_step)):
            if t - B0 < 0:
                continue
            Y = X[(t - B0):t]
            z = self.Z_std(Y)
            if z > maxZ:
                maxZ = z
        return float(maxZ)

    def detect(self, X: np.ndarray, b: float, t_start: int, t_end: int) -> int:
        """Return first t where Z_std(t) > b, else t_end+1."""
        if self._ref_blocks is None:
            raise RuntimeError("Call fit_reference(X0) first")
        X = np.asarray(X, dtype=np.float32)
        T = X.shape[0]
        t_end = min(int(t_end), T)
        B0 = int(self.cfg.B0)

        for t in range(int(t_start), t_end + 1, int(self.cfg.t_step)):
            if t - B0 < 0:
                continue
            Y = X[(t - B0):t]
            if self.Z_std(Y) > float(b):
                return int(t)
        return int(t_end + 1)

# -------------------------
# NN-CUSUM (paper-faithful-ish)
# -------------------------

@dataclass
class NNCusumConfig:
    w: int = 100
    alpha_tr: float = 0.5       # fraction of window used for training
    t_step: int = 1

    # NN hyperparams (same as your fast baseline)
    hidden: Tuple[int, ...] = (16, 16)
    l2: float = 1e-4
    lr: float = 2e-3
    batch_frac: float = 1.0
    epochs_init: int = 10
    epochs_update: int = 2
    shuffle: bool = False

    # early stopping
    use_es: bool = True
    es_rel: float = 0.03
    es_min_epochs: int = 1
    es_norm_ord: int = 2

    N_ref: int = 200
    holdout: bool = True

def _split_holdout(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= 2:
        return x, x
    mid = n // 2
    return x[:mid], x[mid:]

class _TinyMLP:
    """Very small MLP with numpy SGD for speed and determinism."""
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], l2: float, lr: float, seed: int = 0):
        self.in_dim = int(in_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.rng = np.random.default_rng(int(seed))
        dims = (self.in_dim,) + self.hidden + (1,)
        self.W = []
        self.b = []
        for i in range(len(dims) - 1):
            w = self.rng.normal(0, 0.1, size=(dims[i], dims[i+1])).astype(np.float32)
            b = np.zeros((dims[i+1],), dtype=np.float32)
            self.W.append(w)
            self.b.append(b)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = X
        pre = []
        acts = [a]
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            pre.append(z)
            a = self._relu(z)
            acts.append(a)
        zL = a @ self.W[-1] + self.b[-1]
        pre.append(zL)
        acts.append(zL)
        return pre, acts

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        _, acts = self.forward(X)
        return acts[-1].reshape(-1)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_frac: float, shuffle: bool, es: bool,
              es_rel: float, es_min_epochs: int, es_norm_ord: int) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n = X.shape[0]
        if n == 0:
            return
        bs = max(1, int(math.ceil(batch_frac * n)))
        last_W = [w.copy() for w in self.W]
        last_b = [b.copy() for b in self.b]

        for ep in range(int(epochs)):
            if shuffle and n > 1:
                idx = self.rng.permutation(n)
                Xep = X[idx]
                yep = y[idx]
            else:
                Xep, yep = X, y

            # one mini-batch (full-batch by default)
            Xb = Xep[:bs]
            yb = yep[:bs]

            # forward
            pre, acts = self.forward(Xb)
            logit = acts[-1].reshape(-1)
            p = self._sigmoid(logit)

            # grad on logits
            dlogit = (p - yb) / max(1, yb.size)  # BCE derivative for logits
            dlogit = dlogit.reshape(-1, 1)

            # backprop
            dA = dlogit
            for i in reversed(range(len(self.W))):
                Ai = acts[i]
                # grads
                gW = Ai.T @ dA + self.l2 * self.W[i]
                gb = np.sum(dA, axis=0)
                # update
                self.W[i] -= self.lr * gW.astype(np.float32)
                self.b[i] -= self.lr * gb.astype(np.float32)

                if i > 0:
                    dZ = dA @ self.W[i].T
                    dA = dZ * (pre[i-1] > 0).astype(np.float32)

            # early stopping check
            if es and (ep + 1) >= es_min_epochs:
                num = 0.0
                den = 0.0
                for w_old, w_new in zip(last_W, self.W):
                    num += float(np.linalg.norm(w_new - w_old, ord=es_norm_ord))
                    den += float(np.linalg.norm(w_old, ord=es_norm_ord) + 1e-12)
                rel = num / den if den > 0 else 0.0
                last_W = [w.copy() for w in self.W]
                last_b = [b.copy() for b in self.b]
                if rel < es_rel:
                    break

class NNCusumDetector:
    """NN-CUSUM detector.

    Provides `eta_series` to support paper-style alpha calibration:
        Type-I is approximated via P0(max_{t<=k} S_t > b).
    """

    def __init__(self, cfg: Optional[NNCusumConfig] = None, seed: int = 0):
        self.cfg = cfg if cfg is not None else NNCusumConfig()
        self.seed = int(seed)
        self._X0: Optional[np.ndarray] = None
        self._model_prev: Optional[_TinyMLP] = None

    def fit_reference(self, X0: np.ndarray) -> None:
        X0 = np.asarray(X0, dtype=np.float32)
        if X0.ndim != 2:
            raise ValueError("X0 must be 2D (n,dim)")
        self._X0 = X0
        self._model_prev = None

    def _sample_ref(self, n: int, rng: np.random.Generator) -> np.ndarray:
        assert self._X0 is not None
        n0 = self._X0.shape[0]
        if n <= 0:
            return self._X0[:0]
        if n > n0:
            idx = rng.integers(0, n0, size=n)
        else:
            idx = rng.choice(n0, size=n, replace=False)
        return self._X0[idx]

    def _eta_at(self, X_window: np.ndarray, rng: np.random.Generator) -> float:
        """Compute eta_t for window X_window (shape (w,dim))."""
        cfg = self.cfg
        w, dim = X_window.shape
        n_tr = max(2, int(round(cfg.alpha_tr * w)))
        X_ref = self._sample_ref(cfg.N_ref, rng)

        # split into (train,test) inside window and reference
        X1_tr_raw, X1_te_raw = X_window[:n_tr], X_window[n_tr:]
        X0_tr_raw, X0_te_raw = X_ref[:n_tr], X_ref[n_tr:]

        if cfg.holdout:
            X1_tr, X1_te = _split_holdout(X1_tr_raw)
            X0_tr, X0_te2 = _split_holdout(X0_tr_raw)
            X1_test = X1_te_raw if X1_te_raw.shape[0] > 0 else X1_te
            X0_test = X0_te_raw if X0_te_raw.shape[0] > 0 else X0_te2
        else:
            X1_tr, X0_tr = X1_tr_raw, X0_tr_raw
            X1_test, X0_test = X1_te_raw, X0_te_raw

        # training data (binary classification: window=1, ref=0)
        Xtr = np.vstack([X1_tr, X0_tr]).astype(np.float32)
        ytr = np.concatenate([np.ones((X1_tr.shape[0],), dtype=np.float32),
                              np.zeros((X0_tr.shape[0],), dtype=np.float32)], axis=0)

        # initialize / warm start
        if self._model_prev is None:
            model = _TinyMLP(in_dim=dim, hidden=cfg.hidden, l2=cfg.l2, lr=cfg.lr, seed=self.seed + 17)
            epochs = int(cfg.epochs_init)
        else:
            model = self._model_prev
            epochs = int(cfg.epochs_update)

        model.train(
            Xtr, ytr, epochs=epochs,
            batch_frac=float(cfg.batch_frac),
            shuffle=bool(cfg.shuffle),
            es=bool(cfg.use_es),
            es_rel=float(cfg.es_rel),
            es_min_epochs=int(cfg.es_min_epochs),
            es_norm_ord=int(cfg.es_norm_ord),
        )
        self._model_prev = model

        # eta_t: difference of mean scores on test splits
        s1 = model.predict_logit(X1_test)
        s0 = model.predict_logit(X0_test)
        return float(np.mean(s1) - np.mean(s0))

    def eta_series(self, X: np.ndarray, t_start: int, t_end: int) -> Tuple[List[int], List[float]]:
        """Compute eta_t over the horizon (t_start..t_end)."""
        if self._X0 is None:
            raise RuntimeError("Call fit_reference(X0) before eta_series")

        X = np.asarray(X, dtype=np.float32)
        T = X.shape[0]
        t_end = min(int(t_end), T)
        w = int(self.cfg.w)

        rng = np.random.default_rng(self.seed + 999)
        ts: List[int] = []
        etas: List[float] = []

        for t in range(int(t_start), t_end + 1, int(self.cfg.t_step)):
            if t - w < 0:
                continue
            Xw = X[(t - w):t]
            eta = self._eta_at(Xw, rng)
            ts.append(int(t))
            etas.append(float(eta))
        return ts, etas

    @staticmethod
    def max_cusum_from_etas(etas: List[float], D: float) -> float:
        """Compute max CUSUM value along the path given etas and drift D."""
        S = 0.0
        maxS = 0.0
        for eta in etas:
            S = max(S + (float(eta) - float(D)), 0.0)
            if S > maxS:
                maxS = S
        return float(maxS)

    def detect(self, X: np.ndarray, b: float, D: float, t_start: int, t_end: int) -> int:
        """Return first t where CUSUM S_t > b, else t_end+1."""
        if self._X0 is None:
            raise RuntimeError("Call fit_reference(X0) before detect")

        X = np.asarray(X, dtype=np.float32)
        T = X.shape[0]
        t_end = min(int(t_end), T)
        w = int(self.cfg.w)

        rng = np.random.default_rng(self.seed + 999)
        S = 0.0

        for t in range(int(t_start), t_end + 1, int(self.cfg.t_step)):
            if t - w < 0:
                continue
            Xw = X[(t - w):t]
            eta = self._eta_at(Xw, rng)
            S = max(S + (eta - float(D)), 0.0)
            if S > float(b):
                return int(t)
        return int(t_end + 1)
