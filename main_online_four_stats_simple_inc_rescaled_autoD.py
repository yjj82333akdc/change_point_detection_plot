"""
main_online_four_stats_simple.py

A "main_online.py-style" script: edit the USER CONFIG section at the top,
hit Run, and you'll get:

  - one plot shown (optional)
  - the same plot saved to the SAME FOLDER as this script

It plots FOUR online statistics on one graph:

  1) Mode-A statistic: max_j S_{t,j}   (threshold-vector method; mean-inside, max-across t to build h, any-exceed to trigger)
     For plotting, we DO NOT trigger; we just plot the raw max_j S_{t,j}(t).
  2) Mode-B statistic: L2_max(t) = max_m ||g_hat^{(m,n)}||_L2^2 estimated by Monte-Carlo
     (paper-style scan; for plotting we DO NOT trigger)
  3) ScanB statistic (baseline)
  4) NN-CUSUM statistic S_t (baseline)

Notes about thresholds:
- The plotted curves are RAW statistics.
- Mode-A threshold inflation 'a' only affects the alarm threshold (and training-time calibration), not the raw S_{t,j}.
- Mode-B parameters (gamma, C_alpha) are only used to form the theoretical threshold in the paper; the raw L2_max(t)
  does not depend on them. In this script we set C_alpha huge so it never early-stops.

Requirements:
- Put this file in the same folder as:
    threshold_vector_experiment_fast_warm_modes.py
    pdf_baselines_paper_arl_fast.py
    gaussian_mixture.py
- Then run:  python main_online_four_stats_simple.py
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work no matter where you run from
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import threshold_vector_experiment_fast_warm_modes_inc as tv
from gaussian_mixture import gaussian_mixture
from pdf_baselines_paper_arl_fast import (
    ScanBConfig,
    ScanBDetector,
    NNCusumConfig,
    NNCusumDetector,
)

# =========================================================
# USER CONFIG (edit only this block, then click "Run")
# =========================================================

SEED = 123
DIM = 20

# Timeline
N_TRAIN = 500          # reference/baseline length
T_TOTAL = 700          # total length of the generated sequence
CP = 600               # change point (must be > N_TRAIN)

# Choose a scenario by editing generate_sequence() below.
SCENARIO = "mixture"   # "mixture" or "mean_shift" (or write your own)

# -------- Scenario params (mixture) ----------
MEAN_PRE = (2.0, 2.0)
COV_PRE = (1.0, 1.0)
MEAN_POST = (0.0, 0.0)
COV_POST = (1.0, 1.0)

# -------- Scenario params (mean_shift) -------
R_MEANSHIFT = 0.8      # mean shift magnitude used by tv.gen_gaussian_mean_shift

# -------- Mode-A parameters ------------------
J_MIN = -50            # inclusive
J_MAX = -20            # exclusive, python range-style => -50..-21
TV_TRAIN_START = 200   # inside X[:N_TRAIN]
TV_TRAIN_END = 500     # inside X[:N_TRAIN]
T_STEP = 1

# -------- Mode-B parameters ------------------
MC_SAMPLES = 2000      # MC points for L2^2 estimate (paper suggests ~1000)
# gamma/C_alpha only belong to THRESHOLD, not raw stats; keep any values you like
GAMMA = 2.0 / 3.0
C_ALPHA_FOR_PLOTTING = 1e18   # huge => no early stop

# -------- Shared NN hyperparams (Mode-A/B) ---
NN_HIDDEN = (16, 16)   # () for linear/logistic
NN_LR = 2e-3
NN_L2 = 1e-4
EPOCHS_INIT = 20
EPOCHS_UPDATE = 2
BATCH_FRAC = 1.0
SHUFFLE = False
USE_EARLY_STOP = True
N_REF = 200
HOLDOUT = True

# -------- ScanB params -----------------------
SCANB_B0 = 50
SCANB_TSTEP = T_STEP
SCANB_N_BLOCKS = 8
SIGMA_FIT_MAX = 200

# -------- NN-CUSUM params --------------------
NNCUSUM_W = 100
NNCUSUM_TSTEP = T_STEP
# Drift D for CUSUM recursion; if you calibrated D elsewhere, paste it here.
# For simple visualization, 0.0 is OK.
NNCUSUM_D = 0.0

# Automatically estimate the NN-CUSUM drift D from the pre-change baseline of eta_t.
# This is paper-aligned in spirit (D is a calibration object) and helps the CUSUM react post-change.
AUTO_ESTIMATE_D = True
AUTO_D_BASELINE_START_T = N_TRAIN + 1
AUTO_D_BASELINE_END_T = CP - 1
AUTO_D_TRIM_FRAC = 0.0   # 0.0 => plain mean; e.g. 0.05 => 5% trimmed mean

# -------- Rescaling for comparable plot --------
# We rescale ONLY Mode-A and NN-CUSUM to match the baseline mean/std of a reference curve (ScanB or Mode-B).
# This keeps relative spikes comparable without changing Mode-B / ScanB.
RESCALE_MODE_A = True
RESCALE_NNCUSUM = True
RESCALE_REFERENCE = "scanB"   # "scanB" or "modeB"
# Baseline window for estimating scale (typically pre-change):
BASELINE_START_T = N_TRAIN + 1
BASELINE_END_T = CP - 1
RESCALE_EPS = 1e-8

# -------- Plot saving ------------------------
SHOW_PLOT = True
SAVE_PLOT = True
PLOT_FILENAME = "four_stats.png"   # saved into the same folder as this script

# =========================================================
# End USER CONFIG
# =========================================================


def generate_sequence() -> np.ndarray:
    """
    Edit this function if you want a different distribution / sequence.

    Return:
      X: np.ndarray of shape (T_TOTAL, DIM)
    """
    rng = np.random.default_rng(int(SEED))
    np.random.seed(int(SEED))  # gaussian_mixture uses np.random internally

    if SCENARIO == "mixture":
        pre = gaussian_mixture(int(DIM), list(MEAN_PRE), list(COV_PRE))
        post = gaussian_mixture(int(DIM), list(MEAN_POST), list(COV_POST))

        X = pre.generate(int(N_TRAIN))
        for t in range(int(N_TRAIN), int(T_TOTAL)):
            x_new = pre.generate(1) if t < int(CP) else post.generate(1)
            X = np.concatenate([X, x_new], axis=0)
        return np.asarray(X, dtype=np.float32)

    if SCENARIO == "mean_shift":
        # tv generator already returns float32-ish
        return tv.gen_gaussian_mean_shift(
            T=int(T_TOTAL),
            cp=int(CP),
            r=float(R_MEANSHIFT),
            dim=int(DIM),
            seed=int(SEED),
        ).astype(np.float32)

    raise ValueError(f"Unknown SCENARIO={SCENARIO}. Edit generate_sequence() to add your own.")


def compute_mode_a_series(X: np.ndarray) -> Tuple[List[int], List[float]]:
    """
    Mode-A raw series: Smax(t) = max_j S_{t,j}.

    We train the j-models / warm-init cache on X[:N_TRAIN], then run detect_change_mean
    with thresholds set to +inf so it never stops early, returning the full path.
    """
    X0 = X[:int(N_TRAIN)]
    j_list = list(range(int(J_MIN), int(J_MAX)))

    nn_cfg = tv.NNConfig(
        hidden=tuple(int(h) for h in NN_HIDDEN),
        l2=float(NN_L2),
        lr=float(NN_LR),
        epochs_init=int(EPOCHS_INIT),
        epochs_update=int(EPOCHS_UPDATE),
        shuffle=bool(SHUFFLE),
        batch_frac=float(BATCH_FRAC),
        use_es=bool(USE_EARLY_STOP),
        N_ref=int(N_REF),
        holdout=bool(HOLDOUT),
    )

    # train thresholds (and get warm-init models_by_j). 'a' is irrelevant for the raw statistic,
    # but train_threshold_vector needs it to build the threshold vector h.
    j_arr, _M, h, models_by_j = tv.train_threshold_vector(
        X0,
        t_start=int(TV_TRAIN_START),
        t_end=int(TV_TRAIN_END),
        t_step=int(T_STEP),
        j_list=j_list,
        a=1.2,  # any value; plotting uses raw stats only
        nn_cfg=nn_cfg,
        seed=int(SEED + 10),
        return_models=True,
    )

    # For plotting: thresholds = +inf => never triggers, so we get the full path
    h_plot = np.full_like(h, np.inf, dtype=np.float32)

    _alarm_t, df = tv.detect_change_mean(
        X,
        j_arr=j_arr,
        h=h_plot,
        t_start=int(N_TRAIN) + 1,
        t_end=int(T_TOTAL),
        t_step=int(T_STEP),
        nn_cfg=nn_cfg,
        seed=int(SEED + 11),
        return_path=True,
        init_models=models_by_j,
    )
    if df is None:
        raise RuntimeError("Mode-A returned df=None unexpectedly.")
    ts = [int(t) for t in df["t"].tolist()]
    vals = [float(v) for v in df["S_max"].tolist()]
    return ts, vals


def compute_mode_b_series(X: np.ndarray) -> Tuple[List[int], List[float]]:
    """
    Mode-B raw series: L2_max(t) = max_m ||g_hat^{(m,n)}||_L2^2 estimated via MC.

    We set C_alpha huge so the routine never early-stops (alarm logic off).
    """
    nn_cfg = tv.NNConfig(
        hidden=tuple(int(h) for h in NN_HIDDEN),
        l2=float(NN_L2),
        lr=float(NN_LR),
        epochs_init=int(EPOCHS_INIT),
        epochs_update=int(EPOCHS_UPDATE),
        shuffle=bool(SHUFFLE),
        batch_frac=float(BATCH_FRAC),
        use_es=bool(USE_EARLY_STOP),
        N_ref=int(N_REF),
        holdout=bool(HOLDOUT),
    )

    _alarm_t, df = tv.detect_change_l2_paper(
        X,
        N=int(N_TRAIN),
        gamma=float(GAMMA),
        C_alpha=float(C_ALPHA_FOR_PLOTTING),
        mc_samples=int(MC_SAMPLES),
        t_step=int(T_STEP),
        t_start=int(N_TRAIN) + 1,
        t_end=int(T_TOTAL),
        nn_cfg=nn_cfg,
        seed=int(SEED + 12),
        return_path=True,
    )
    if df is None:
        raise RuntimeError("Mode-B returned df=None unexpectedly.")
    ts = [int(t) for t in df["t"].tolist()]
    vals = [float(v) for v in df["L2_max"].tolist()]
    return ts, vals


def compute_scanb_series(X: np.ndarray) -> Tuple[List[int], List[float]]:
    X0 = X[:int(N_TRAIN)]
    cfg = ScanBConfig(B0=int(SCANB_B0), N_blocks=int(SCANB_N_BLOCKS), t_step=int(SCANB_TSTEP), sigma=None)

    det = ScanBDetector(cfg=cfg, seed=int(SEED + 20))
    det.fit_reference(X0, sigma_fit_max=int(SIGMA_FIT_MAX))

    ts: List[int] = []
    zs: List[float] = []
    B0 = int(cfg.B0)
    for t in range(int(N_TRAIN) + 1, int(T_TOTAL) + 1, int(cfg.t_step)):
        if t - B0 < 0:
            continue
        Y = X[(t - B0):t]
        ts.append(int(t))
        zs.append(float(det.Z_std(Y)))
    return ts, zs


def compute_nncusum_series(X: np.ndarray) -> Tuple[List[int], List[float]]:
    X0 = X[:int(N_TRAIN)]
    cfg = NNCusumConfig(
        w=int(NNCUSUM_W),
        alpha_tr=0.5,
        t_step=int(NNCUSUM_TSTEP),
        hidden=tuple(int(h) for h in NN_HIDDEN),
        l2=float(NN_L2),
        lr=float(NN_LR),
        batch_frac=float(BATCH_FRAC),
        epochs_init=max(1, int(EPOCHS_INIT // 2)),
        epochs_update=max(1, int(EPOCHS_UPDATE)),
        shuffle=bool(SHUFFLE),
        use_es=bool(USE_EARLY_STOP),
        N_ref=int(N_REF),
        holdout=bool(HOLDOUT),
    )

    det = NNCusumDetector(cfg=cfg, seed=int(SEED + 30))
    det.fit_reference(X0)

    times, etas = det.eta_series(X, t_start=int(N_TRAIN) + 1, t_end=int(T_TOTAL))

    # Optionally estimate drift D from a pre-change baseline slice of eta_t.
    D_use = float(NNCUSUM_D)
    if AUTO_ESTIMATE_D:
        t_arr = np.asarray(times, dtype=int)
        e_arr = np.asarray(etas, dtype=float)
        msk = (t_arr >= int(AUTO_D_BASELINE_START_T)) & (t_arr <= int(AUTO_D_BASELINE_END_T)) & np.isfinite(e_arr)
        if msk.sum() >= 20:
            vals = np.sort(e_arr[msk])
            if AUTO_D_TRIM_FRAC > 0.0:
                k = int(len(vals) * float(AUTO_D_TRIM_FRAC))
                if 2 * k < len(vals):
                    vals = vals[k:len(vals)-k]
            D_use = float(np.mean(vals))
            print(f"[nncusum] AUTO D estimated from eta baseline: D={D_use:.6g} using {len(vals)} points")
        else:
            print("[nncusum] AUTO D requested but not enough baseline eta points; using NNCUSUM_D")

    # CUSUM recursion: S_t = max(0, S_{t-1} + eta_t - D)
    S = 0.0
    S_series: List[float] = []
    for eta in etas:
        S = max(0.0, S + float(eta) - float(D_use))
        S_series.append(float(S))

    return [int(t) for t in times], S_series


def _align_on_time(
    tA: List[int], yA: List[float],
    tB: List[int], yB: List[float],
    tS: List[int], yS: List[float],
    tN: List[int], yN: List[float],
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Align by time via dict lookup. Missing values become NaN (matplotlib will break lines).
    """
    dA = {int(t): float(v) for t, v in zip(tA, yA)}
    dB = {int(t): float(v) for t, v in zip(tB, yB)}
    dS = {int(t): float(v) for t, v in zip(tS, yS)}
    dN = {int(t): float(v) for t, v in zip(tN, yN)}

    all_t = sorted(set(dA.keys()) | set(dB.keys()) | set(dS.keys()) | set(dN.keys()))
    yA2, yB2, yS2, yN2 = [], [], [], []
    for t in all_t:
        yA2.append(dA.get(t, float("nan")))
        yB2.append(dB.get(t, float("nan")))
        yS2.append(dS.get(t, float("nan")))
        yN2.append(dN.get(t, float("nan")))
    return all_t, yA2, yB2, yS2, yN2



def _affine_rescale_to_match(y: List[float], y_ref: List[float], t: List[int], *, t0: int, t1: int, eps: float) -> List[float]:
    """
    Affine-rescale y so that on baseline window t in [t0, t1], it matches y_ref's mean/std:
        y' = (y - mu_y)/std_y * std_ref + mu_ref

    NaN/inf are ignored. If std_y is tiny (nearly constant baseline), we do NOT rescale
    to avoid exploding noise.
    """
    y_arr = np.asarray(y, dtype=float)
    y_ref_arr = np.asarray(y_ref, dtype=float)
    t_arr = np.asarray(t, dtype=int)

    mask = (t_arr >= int(t0)) & (t_arr <= int(t1)) & np.isfinite(y_arr) & np.isfinite(y_ref_arr)
    if mask.sum() < 20:
        return y

    mu_y = float(np.nanmean(y_arr[mask]))
    sd_y = float(np.nanstd(y_arr[mask]))
    mu_r = float(np.nanmean(y_ref_arr[mask]))
    sd_r = float(np.nanstd(y_ref_arr[mask]))

    if sd_y <= eps * 10 or sd_r <= eps * 10:
        return y

    y_scaled = (y_arr - mu_y) / sd_y * sd_r + mu_r
    return [float(v) for v in y_scaled]


def main() -> None:
    if not (0 < N_TRAIN < T_TOTAL):
        raise ValueError("Need 0 < N_TRAIN < T_TOTAL.")
    if not (N_TRAIN < CP < T_TOTAL):
        raise ValueError("Need N_TRAIN < CP < T_TOTAL.")

    X = generate_sequence()
    if X.shape != (int(T_TOTAL), int(DIM)):
        # allow user-defined generator to return different shape, but warn clearly
        raise ValueError(f"generate_sequence() returned shape {X.shape}, expected ({T_TOTAL},{DIM}).")

    print(f"[data] X shape={X.shape}, N_TRAIN={N_TRAIN}, CP={CP}")

    print("[Mode-A] computing series...")
    tA, yA = compute_mode_a_series(X)

    print("[Mode-B] computing series (may be slow if T_TOTAL is large)...")
    tB, yB = compute_mode_b_series(X)

    print("[ScanB] computing series...")
    tS, yS = compute_scanb_series(X)

    print("[NN-CUSUM] computing series...")
    tN, yN = compute_nncusum_series(X)

    t, yA2, yB2, yS2, yN2 = _align_on_time(tA, yA, tB, yB, tS, yS, tN, yN)

    
    # ----------------------------
    # Optional: rescale Mode-A and/or NN-CUSUM to make the four curves comparable on one vertical scale.
    # We keep Mode-B and ScanB unchanged, and affine-transform the selected curves to match the baseline mean/std
    # of the reference curve (ScanB by default).
    # ----------------------------
    if RESCALE_REFERENCE.lower() == "modeb":
        ref_curve = yB2
        ref_name = "Mode-B"
    else:
        ref_curve = yS2
        ref_name = "ScanB"

    if RESCALE_MODE_A:
        yA2 = _affine_rescale_to_match(yA2, ref_curve, t, t0=BASELINE_START_T, t1=BASELINE_END_T, eps=RESCALE_EPS)
        print(f"[rescale] Mode-A rescaled to match {ref_name} on baseline t in [{BASELINE_START_T},{BASELINE_END_T}]")

    if RESCALE_NNCUSUM:
        yN2 = _affine_rescale_to_match(yN2, ref_curve, t, t0=BASELINE_START_T, t1=BASELINE_END_T, eps=RESCALE_EPS)
        print(f"[rescale] NN-CUSUM rescaled to match {ref_name} on baseline t in [{BASELINE_START_T},{BASELINE_END_T}]")
    plt.figure(figsize=(10, 5))
    plt.plot(t, yA2, label="Mode-A: max_j ΔS_{t,j} (max increment)")
    plt.plot(t, yB2, label="Mode-B: L2_max(t) (MC)")
    plt.plot(t, yS2, label="ScanB statistic")
    plt.plot(t, yN2, label="NN-CUSUM statistic S_t")

    plt.axvline(int(N_TRAIN), linestyle="--", linewidth=1.0, label="N_train")
    plt.axvline(int(CP), linestyle="--", linewidth=1.0, label="change point")

    plt.title(f"Four online statistics (scenario={SCENARIO}, dim={DIM})")
    plt.xlabel("t")
    plt.ylabel("statistic value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOT:
        out_path = os.path.join(THIS_DIR, str(PLOT_FILENAME))
        plt.savefig(out_path, dpi=160)
        print(f"[saved] {out_path}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
