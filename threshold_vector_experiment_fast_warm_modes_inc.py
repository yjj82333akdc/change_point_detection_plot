#!/usr/bin/env python3
"""
threshold_vector_experiment_fast_warm.py  (MODE-A + MODE-B)

MODE-A ("mean"): your original "threshold vector" method with tunable negative offsets j
    - You provide j_list, e.g. range(-50, -20) => 30 coordinates.
    - For each (t, j), split a fixed 200-length prefix window into B and A:
          B = X[t-200 : t+j],   A = X[t+j : t]
      train a classifier (warm-started per j across t), and score:
          S_{t,j} = mean( f_theta(score_block) )
      trigger if any coordinate exceeds its threshold h_j.

MODE-B ("l2"): paper Algorithm-1 style L2 scan, with (m,n) as *dummy scan variables*
    - Fix N (training length). For each t = N+1, N+2, ...
      scan all m in {N, ..., t-1} and set n = t - m.
      Build two samples:
          class-0: X[0:m],  class-1: X[m:t]
      Train/update a classifier and compute Monte-Carlo estimate of L2-norm squared:
          ||g_hat^{(m,n)}||_L2^2 ≈ mean_k f_theta(Z_k)^2
      where Z_k are sampled from the pooled empirical distribution of X[0:t].
      Trigger if for any m:
          ||g_hat||_L2^2 >= C_alpha*(log(n)^4 / n^gamma + log(t)/n).

Warm-start acceleration:
    - MODE-A: one model per j (as before).
    - MODE-B: one model per m, updated across t using init_from=previous_model(m).

Notes:
    - For MODE-B, the full scan over all m is O(t^2) work (as in the pseudocode).
      Warm-start helps, but this is still expensive for very large t.

This file depends on NN_warm_init.NumpyLogisticNN.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from NN_warm_init import NumpyLogisticNN


# ----------------------------
# Utilities
# ----------------------------
def _split_holdout(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a window into (train, score) halves."""
    w = x.shape[0]
    mid = max(1, w // 2)
    return x[:mid], x[mid:]


def gen_gaussian_mean_shift(
    T: int,
    cp: Optional[int],
    r: float,
    dim: int,
    mu0: float = 0.0,
    mu1: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """X_t ~ N(mu0, r^2 I) for t < cp, and N(mu1, r^2 I) for t >= cp. If cp=None => no change."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=mu0, scale=r, size=(T, dim)).astype(np.float32)
    if cp is not None:
        X[cp:, :] += (mu1 - mu0)
    return X


@dataclass
class NNConfig:
    hidden: Tuple[int, ...] = (16, 16)
    l2: float = 1e-4
    lr: float = 2e-3

    epochs_init: int = 20     # first time for each j (MODE-A) / each m (MODE-B)
    epochs_update: int = 2    # subsequent updates

    shuffle: bool = False
    batch_frac: float = 1.0   # batch_size ~ batch_frac * (#samples)

    # drift-based early stop (optional)
    use_es: bool = True
    es_rel: float = 0.03
    es_min_epochs: int = 1
    es_norm_ord: int = 2
    N_ref: int = 200          # size of reference prefix X_ref for drift stop

    # variance control
    holdout: bool = True


def _batch_size(n: int, cfg: NNConfig) -> int:
    return max(16, int(cfg.batch_frac * max(1, n)))


def _fit_one(
    model: NumpyLogisticNN,
    B_tr: np.ndarray,
    A_tr: np.ndarray,
    X_ref: Optional[np.ndarray],
    epochs: int,
    cfg: NNConfig,
) -> None:
    fit_kwargs = dict(
        epochs=int(epochs),
        batch_size=_batch_size(B_tr.shape[0] + A_tr.shape[0], cfg),
        lr=float(cfg.lr),
        verbose=0,
        shuffle=bool(cfg.shuffle),
    )
    if cfg.use_es and X_ref is not None and len(X_ref) > 0:
        fit_kwargs.update(
            X_test=X_ref,
            es_rel=float(cfg.es_rel),
            es_min_epochs=int(min(cfg.es_min_epochs, epochs)),
            es_norm_ord=int(cfg.es_norm_ord),
        )
    model.fit(B_tr, A_tr, **fit_kwargs)


# ----------------------------
# MODE-A: threshold vector training
# ----------------------------
def train_threshold_vector(
    X0: np.ndarray,
    t_start: int = 200,
    t_end: int = 500,
    t_step: int = 1,
    j_list: Optional[List[int]] = None,
    a: float = 1.2,
    nn_cfg: Optional[NNConfig] = None,
    seed: int = 123,
    return_models: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, NumpyLogisticNN]]]:
    """
    Train increment-thresholds on X0: M_j = max_t (S_{t,j}-S_{t-1,j}), then h_j = a * M_j.

    Returns:
      j_arr: (m,) array of j values
      M:     (m,) max-over-t values
      h:     (m,) thresholds
      models_by_j (optional): dict of last models per j for warm-start in detection
    """
    if nn_cfg is None:
        nn_cfg = NNConfig()

    if j_list is None:
        j_list = list(range(-50, -20))  # -50..-21 inclusive
    j_arr = np.array(j_list, dtype=int)
    m = len(j_arr)

    n0, dim = X0.shape
    if t_end > n0:
        raise ValueError(f"Need X0 length >= t_end (got {n0}, t_end={t_end}).")
    if t_start < 200:
        raise ValueError("t_start must be >= 200 because this mode uses X[t-200:...] windows.")

    # Reference set for drift-based early stop: prefix from training region
    X_ref = X0[: min(nn_cfg.N_ref, n0)] if nn_cfg.use_es else None

    ts = np.arange(t_start, t_end + 1, t_step, dtype=int)
    S = np.full((len(ts), m), np.nan, dtype=np.float32)

    rng = np.random.default_rng(seed)
    models_by_j: Dict[int, NumpyLogisticNN] = {}

    # Warm-start: one model per j, updated as t increases (j outer loop).
    for jj, j in enumerate(j_arr):
        model_prev: Optional[NumpyLogisticNN] = None

        for it, t in enumerate(ts):
            B = X0[(t - 200):(t + j)]
            A = X0[(t + j):t]

            if B.shape[0] <= 1 or A.shape[0] <= 1:
                continue

            if nn_cfg.holdout:
                B_tr, B_sc = _split_holdout(B)
                A_tr, _ = _split_holdout(A)
                score_block = B_sc
            else:
                B_tr, A_tr = B, A
                score_block = B

            model = NumpyLogisticNN(
                input_dim=dim,
                hidden=nn_cfg.hidden,
                l2=nn_cfg.l2,
                seed=int(rng.integers(0, 10_000_000)),
                init_from=model_prev,
            )
            epochs = nn_cfg.epochs_init if model_prev is None else nn_cfg.epochs_update
            _fit_one(model, B_tr, A_tr, X_ref, epochs, nn_cfg)

            S[it, jj] = float(np.mean(model.f_theta(score_block)))
            model_prev = model

        if model_prev is not None:
            models_by_j[int(j)] = model_prev
    # By default, we calibrate thresholds on *one-step increments*:
    #   dS_{t,j} = S_{t,j} - S_{t-1,j}
    # and set M_j = max_t dS_{t,j}, h_j = a * M_j.
    #
    # This matches the "max increment" Mode-A variant.
    dS = np.full_like(S, np.nan, dtype=np.float32)
    for it in range(1, len(ts)):
        prev = S[it - 1]
        cur = S[it]
        mask = np.isfinite(prev) & np.isfinite(cur)
        dS[it, mask] = cur[mask] - prev[mask]

    # Level maxima (reference only, not returned):
    # M_level = np.nanmax(S, axis=0).astype(np.float64)

    M = np.nanmax(dS, axis=0).astype(np.float64)
    h = (a * M).astype(np.float64)

    if return_models:
        return j_arr.astype(int), M, h, models_by_j
    return j_arr.astype(int), M, h


# ----------------------------
# MODE-A: detection (threshold vector)
# ----------------------------
def detect_change_mean(
    X: np.ndarray,
    h: np.ndarray,
    j_arr: np.ndarray,
    t_step: int = 1,
    t_start: int = 501,
    t_end: int = 1000,
    nn_cfg: Optional[NNConfig] = None,
    seed: int = 456,
    return_path: bool = True,
    init_models: Optional[Dict[int, NumpyLogisticNN]] = None,
) -> Tuple[int, Optional[pd.DataFrame]]:
    """
    Online scan in [t_start, t_end]. At each t, compute S_{t,j} for all j and alarm if any S_{t,j} > h_j.

    Returns:
      alarm_t: first t where alarm happens, else (t_end+1)
      df_path: optional DataFrame with columns:
               t, S_max (= max_j increment dS_{t,j}), S_level_max (= max_j level S_{t,j}), j_star, exceed_any,
               and optionally all level S_{t,j} and increments dS_{t,j}.
    """
    if nn_cfg is None:
        nn_cfg = NNConfig()

    T, dim = X.shape
    t_end = min(t_end, T)

    rng = np.random.default_rng(seed)

    # Warm-start per j. Optionally seed from training-phase models.
    if init_models is None:
        models: Dict[int, Optional[NumpyLogisticNN]] = {int(j): None for j in j_arr}
    else:
        models = {int(j): init_models.get(int(j), None) for j in j_arr}

    X_ref = X[: min(nn_cfg.N_ref, T)] if nn_cfg.use_es else None

    rows = []
    prev_s_vec: Optional[np.ndarray] = None
    for t in range(int(t_start), int(t_end) + 1, int(t_step)):
        s_vec = np.full((len(j_arr),), np.nan, dtype=np.float64)

        for idx, j in enumerate(j_arr):
            j = int(j)
            B = X[(t - 200):(t + j)]
            A = X[(t + j):t]
            if B.shape[0] <= 1 or A.shape[0] <= 1:
                continue

            if nn_cfg.holdout:
                B_tr, B_sc = _split_holdout(B)
                A_tr, _ = _split_holdout(A)
                score_block = B_sc
            else:
                B_tr, A_tr = B, A
                score_block = B

            init_from = models[j]
            model = NumpyLogisticNN(
                input_dim=dim,
                hidden=nn_cfg.hidden,
                l2=nn_cfg.l2,
                seed=int(rng.integers(0, 10_000_000)),
                init_from=init_from,
            )
            epochs = nn_cfg.epochs_init if init_from is None else nn_cfg.epochs_update
            _fit_one(model, B_tr, A_tr, X_ref, epochs, nn_cfg)
            models[j] = model

            s_vec[idx] = float(np.mean(model.f_theta(score_block)))
        # One-step increments (aligned with t_step):
        #   dS_{t,j} := S_{t,j} - S_{t-t_step,j}
        dS_vec = np.full_like(s_vec, np.nan, dtype=np.float64)
        if prev_s_vec is not None:
            mask = np.isfinite(prev_s_vec) & np.isfinite(s_vec)
            dS_vec[mask] = s_vec[mask] - prev_s_vec[mask]

        # Update memory for next step
        prev_s_vec = s_vec.copy()

        # Alarm logic is based on increments
        exceed = np.isfinite(dS_vec) & (dS_vec > h)
        exceed_any = bool(np.any(exceed))

        # Scalar summaries:
        #   S_max := max_j dS_{t,j}  (max increment across components)
        #   S_level_max := max_j S_{t,j} (level max; kept for reference)
        if np.any(np.isfinite(dS_vec)):
            j_star = int(j_arr[int(np.nanargmax(dS_vec))])
            s_max = float(np.nanmax(dS_vec))
        else:
            j_star = 999999
            s_max = float("nan")

        if np.any(np.isfinite(s_vec)):
            s_level_max = float(np.nanmax(s_vec))
        else:
            s_level_max = float("nan")
        row = {
            "t": int(t),
            "S_max": s_max,
            "S_level_max": s_level_max,
            "j_star": j_star,
            "exceed_any": exceed_any,
        }
        if return_path:
            for idx, j in enumerate(j_arr):
                j_int = int(j)
                row[f"S_j{j_int}"] = float(s_vec[idx]) if np.isfinite(s_vec[idx]) else np.nan
                row[f"dS_j{j_int}"] = float(dS_vec[idx]) if np.isfinite(dS_vec[idx]) else np.nan
        rows.append(row)

        if exceed_any:
            df = pd.DataFrame(rows) if return_path else None
            return int(t), df

    df = pd.DataFrame(rows) if return_path else None
    return int(t_end) + 1, df


# ----------------------------
# MODE-B: paper Algorithm-1 L2 scan with (m,n) dummy indices
# ----------------------------
def detect_change_l2_paper(
    X: np.ndarray,
    N: int,
    gamma: float = 2.0 / 3.0,
    C_alpha: float = 1.0,
    mc_samples: int = 1000,
    t_step: int = 1,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
    nn_cfg: Optional[NNConfig] = None,
    seed: int = 456,
    return_path: bool = True,
) -> Tuple[int, Optional[pd.DataFrame]]:
    """
    Paper-style scan: for each t >= N+1, scan all m in [N, t-1], set n = t-m, and alarm if
      ||g_hat^{(m,n)}||_L2^2 >= C_alpha*(log(n)^4 / n^gamma + log(t)/n).

    Here we approximate ||g_hat||_L2^2 with Monte-Carlo under the empirical measure on X[0:t]:
      Z_k ~ Uniform({X_0,...,X_{t-1}}),  ||g_hat||_L2^2 ≈ mean_k f_theta(Z_k)^2.

    Warm-start: cache one model per m and update it across t with init_from=cache[m].

    Practical note: we skip m where either class has <2 samples (to avoid degenerate training).
    """
    if nn_cfg is None:
        nn_cfg = NNConfig()

    T, dim = X.shape
    if t_end is None:
        t_end = T
    t_end = min(int(t_end), T)

    if t_start is None:
        t_start = N + 1
    t_start = max(int(t_start), N + 1)

    rng = np.random.default_rng(seed)

    # Reference for drift ES: only from the initial training region
    X_ref = X[: min(nn_cfg.N_ref, max(N, 1))] if nn_cfg.use_es else None

    # Warm-init cache per m
    cache_by_m: Dict[int, Optional[NumpyLogisticNN]] = {}

    rows = []
    for t in range(int(t_start), int(t_end) + 1, int(t_step)):
        # MC samples from pooled X[:t]
        if t <= 1:
            continue
        idxs = rng.integers(0, t, size=int(mc_samples))
        Z = X[idxs]

        l2_max = -np.inf
        m_star = None
        n_star = None
        thr_star = None
        exceed_any = False

        # Scan all m (dummy variable)
        for m in range(int(N), int(t)):
            n = int(t - m)
            if m < 2 or n < 2:
                continue

            B = X[:m]
            A = X[m:t]

            if nn_cfg.holdout:
                B_tr, _ = _split_holdout(B)
                A_tr, _ = _split_holdout(A)
            else:
                B_tr, A_tr = B, A

            init_from = cache_by_m.get(m, None)
            model = NumpyLogisticNN(
                input_dim=dim,
                hidden=nn_cfg.hidden,
                l2=nn_cfg.l2,
                seed=int(rng.integers(0, 10_000_000)),
                init_from=init_from,
            )
            epochs = nn_cfg.epochs_init if init_from is None else nn_cfg.epochs_update
            _fit_one(model, B_tr, A_tr, X_ref, epochs, nn_cfg)
            cache_by_m[m] = model

            # L2^2 via MC (empirical)
            g = model.f_theta(Z).astype(np.float64)
            l2_sq = float(np.mean(g * g))

            # Threshold depends on n and t
            # (natural logs; consistent with the paper up to constants)
            thr = float(C_alpha * ((np.log(max(n, 2.0)) ** 4) / (n ** float(gamma)) + np.log(max(t, 2.0)) / n))

            if l2_sq > l2_max:
                l2_max = l2_sq
                m_star = int(m)
                n_star = int(n)
                thr_star = thr

            if l2_sq >= thr:
                exceed_any = True
                # paper says "ALARM <- TRUE; break" inside m loop
                break

        row = {
            "t": int(t),
            "L2_max": float(l2_max) if np.isfinite(l2_max) else np.nan,
            "m_star": int(m_star) if m_star is not None else -1,
            "n_star": int(n_star) if n_star is not None else -1,
            "thr_star": float(thr_star) if thr_star is not None else np.nan,
            "exceed_any": bool(exceed_any),
        }
        rows.append(row)

        if exceed_any:
            df = pd.DataFrame(rows) if return_path else None
            return int(t), df

    df = pd.DataFrame(rows) if return_path else None
    return int(t_end) + 1, df


# ----------------------------
# Unified entrypoint (two modes)
# ----------------------------
def detect_change(
    *,
    mode: str,
    X: np.ndarray,
    # MODE-A args
    h: Optional[np.ndarray] = None,
    j_arr: Optional[np.ndarray] = None,
    init_models: Optional[Dict[int, NumpyLogisticNN]] = None,
    # MODE-B args
    N: int = 500,
    gamma: float = 2.0 / 3.0,
    C_alpha: float = 1.0,
    mc_samples: int = 1000,
    # shared
    t_step: int = 1,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
    nn_cfg: Optional[NNConfig] = None,
    seed: int = 456,
    return_path: bool = True,
) -> Tuple[int, Optional[pd.DataFrame]]:
    mode = str(mode).lower().strip()
    if mode in ("mean", "mode-a", "a"):
        if h is None or j_arr is None:
            raise ValueError("MODE-A requires h and j_arr.")
        return detect_change_mean(
            X=X,
            h=h,
            j_arr=j_arr,
            t_step=t_step,
            t_start=501 if t_start is None else int(t_start),
            t_end=1000 if t_end is None else int(t_end),
            nn_cfg=nn_cfg,
            seed=seed,
            return_path=return_path,
            init_models=init_models,
        )
    if mode in ("l2", "mode-b", "b"):
        return detect_change_l2_paper(
            X=X,
            N=int(N),
            gamma=float(gamma),
            C_alpha=float(C_alpha),
            mc_samples=int(mc_samples),
            t_step=t_step,
            t_start=t_start,
            t_end=t_end,
            nn_cfg=nn_cfg,
            seed=seed,
            return_path=return_path,
        )
    raise ValueError(f"Unknown mode={mode}. Use 'mean' (MODE-A) or 'l2' (MODE-B).")


# ----------------------------
# CLI
# ----------------------------
def parse_hidden(s: str, linear: bool) -> Tuple[int, ...]:
    if linear:
        return ()
    s = s.strip()
    if not s:
        return (16, 16)
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="mean", choices=["mean", "l2"], help="mean=MODE-A threshold-vector; l2=MODE-B paper scan")

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--r", type=float, default=1.0)

    ap.add_argument("--cp", type=int, default=800, help="change point; set -1 to randomize in (N+1..T-1)")
    ap.add_argument("--seed", type=int, default=0, help="seed for data generation")

    # MODE-A threshold inflation
    ap.add_argument("--a", type=float, default=1.2, help="MODE-A: threshold inflation coefficient")

    # training length N (also the paper's N for MODE-B)
    ap.add_argument("--N", type=int, default=500, help="baseline length N; MODE-A uses X[:N] to train thresholds; MODE-B starts scanning at t=N+1")

    # MODE-B parameters
    ap.add_argument("--gamma", type=float, default=2.0 / 3.0, help="MODE-B: tuning parameter gamma")
    ap.add_argument("--C-alpha", dest="C_alpha", type=float, default=1.0, help="MODE-B: threshold constant C_alpha")
    ap.add_argument("--mc-samples", type=int, default=1000, help="MODE-B: MC samples to estimate L2^2")

    # MODE-A threshold-training time range within X[:N]
    ap.add_argument("--tstart", type=int, default=200)
    ap.add_argument("--tend", type=int, default=500)
    ap.add_argument("--jmin", type=int, default=-50, help="MODE-A: minimum j (inclusive)")
    ap.add_argument("--jmax", type=int, default=-20, help="MODE-A: maximum j (exclusive), e.g. -20 gives -50..-21")

    # NN hyperparams
    ap.add_argument("--hidden", type=str, default="16,16")
    ap.add_argument("--linear", action="store_true", help="use logistic regression (hidden=())")
    ap.add_argument("--epochs-init", type=int, default=20)
    ap.add_argument("--epochs-update", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch-frac", type=float, default=0.25)

    # early stop
    ap.add_argument("--no-es", action="store_true")
    ap.add_argument("--es-rel", type=float, default=0.03)
    ap.add_argument("--es-min-epochs", type=int, default=1)
    ap.add_argument("--es-norm-ord", type=int, default=2)
    ap.add_argument("--N-ref", type=int, default=200)

    # variance control
    ap.add_argument("--no-holdout", action="store_true")

    # detection phase
    ap.add_argument("--detect", action="store_true", help="run detection on full sequence")
    ap.add_argument("--detect-tstart", type=int, default=None)
    ap.add_argument("--detect-tend", type=int, default=None)
    ap.add_argument("--t-step", type=int, default=1)

    # outputs
    ap.add_argument("--out-thresh", type=str, default="threshold_vector.csv")
    ap.add_argument("--out-path", type=str, default="scan_path.csv")

    args = ap.parse_args()

    # Change point
    if args.cp == -1:
        rng = np.random.default_rng(args.seed + 999)
        cp = int(rng.integers(args.N + 1, min(args.T - 1, args.T - 1) + 1))
    else:
        cp = int(args.cp)

    X = gen_gaussian_mean_shift(T=args.T, cp=cp, r=args.r, dim=args.dim, seed=args.seed)

    nn_cfg = NNConfig(
        hidden=parse_hidden(args.hidden, args.linear),
        l2=float(args.l2),
        lr=float(args.lr),
        epochs_init=max(1, int(args.epochs_init)),
        epochs_update=max(1, int(args.epochs_update)),
        batch_frac=float(args.batch_frac),
        use_es=not bool(args.no_es),
        es_rel=float(args.es_rel),
        es_min_epochs=max(1, int(args.es_min_epochs)),
        es_norm_ord=int(args.es_norm_ord),
        N_ref=max(10, int(args.N_ref)),
        holdout=not bool(args.no_holdout),
    )

    print(f"Mode: {args.mode} | ground-truth cp={cp}")

    # MODE-A: train threshold vector on X[:N]
    if args.mode == "mean":
        X0 = X[: int(args.N)]
        j_list = list(range(int(args.jmin), int(args.jmax)))
        j_arr, M, h, models_by_j = train_threshold_vector(
            X0=X0,
            t_start=int(args.tstart),
            t_end=int(args.tend),
            t_step=1,
            j_list=j_list,
            a=float(args.a),
            nn_cfg=nn_cfg,
            seed=123,
            return_models=True,
        )
        df_thresh = pd.DataFrame({"j": j_arr, "M_max_over_t": M, "h": h})
        df_thresh.to_csv(args.out_thresh, index=False)
        print("\n=== MODE-A: Threshold vector trained ===")
        print(df_thresh.to_string(index=False))
        print(f"Saved thresholds to: {args.out_thresh}")

        if args.detect:
            alarm_t, df_path = detect_change(
                mode="mean",
                X=X,
                h=h,
                j_arr=j_arr,
                init_models=models_by_j,
                t_step=max(1, int(args.t_step)),
                t_start=(args.detect_tstart if args.detect_tstart is not None else 501),
                t_end=(args.detect_tend if args.detect_tend is not None else args.T),
                nn_cfg=nn_cfg,
                seed=456,
                return_path=True,
            )
            print("\n=== MODE-A Detection ===")
            if alarm_t <= (args.detect_tend if args.detect_tend is not None else args.T):
                print(f"ALARM at t={alarm_t} (exists j: S_{t,j} > h_j)")
                print(f"Delay (relative to cp): {max(alarm_t - cp, 0)}")
            else:
                print(f"No alarm up to t={(args.detect_tend if args.detect_tend is not None else args.T)}.")

            if df_path is not None:
                df_path.to_csv(args.out_path, index=False)
                print(f"Saved scan path to: {args.out_path}")

        return

    # MODE-B: paper scan (no threshold vector training)
    if args.mode == "l2":
        if not args.detect:
            print("MODE-B (l2) requires --detect to run the scan.")
            return

        alarm_t, df_path = detect_change(
            mode="l2",
            X=X,
            N=int(args.N),
            gamma=float(args.gamma),
            C_alpha=float(args.C_alpha),
            mc_samples=int(args.mc_samples),
            t_step=max(1, int(args.t_step)),
            t_start=(args.detect_tstart if args.detect_tstart is not None else int(args.N) + 1),
            t_end=(args.detect_tend if args.detect_tend is not None else args.T),
            nn_cfg=nn_cfg,
            seed=456,
            return_path=True,
        )
        print("\n=== MODE-B (paper L2) Detection ===")
        if alarm_t <= (args.detect_tend if args.detect_tend is not None else args.T):
            print(f"ALARM at t={alarm_t}")
            print(f"Delay (relative to cp): {max(alarm_t - cp, 0)}")
        else:
            print(f"No alarm up to t={(args.detect_tend if args.detect_tend is not None else args.T)}.")

        if df_path is not None:
            df_path.to_csv(args.out_path, index=False)
            print(f"Saved scan path to: {args.out_path}")

        return


if __name__ == "__main__":
    main()
