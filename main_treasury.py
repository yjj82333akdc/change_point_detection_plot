#https://home.treasury.gov/interest-rates-data-csv-archive?utm_source=chatgpt.com


"""

import numpy as np
import csv
import matplotlib.pyplot as plt
path = "/Users/daren/git/NN_change/bill.csv"

# Discover number of columns, then skip the first (date) column
with open(path, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
ncols = len(header)
usecols = tuple(range(1, ncols))  # skip col 0 (date)

A = np.genfromtxt(
    path,
    delimiter=",",
    skip_header=1,
    usecols=usecols,
    dtype=float,
    missing_values=["", "NA", "NaN", "nan", "."],
    filling_values=np.nan,
)

# Drop rows containing any NaN (optional)
A = A[~np.isnan(A).any(axis=1)]

print(A.shape)  # (N, d)


data=A[::-1,]

N_train=100
N_total=data.shape[0]
N_total=300
dim=data.shape[1]
from NN_warm_init import NumpyLogisticNN


#np.linalg.norm(np.mean(A[:N_train-50], axis=0) - np.mean(A[N_train-50:N_train], axis=0))

#np.linalg.norm(np.mean(A[:200], axis=0) - np.mean(A[200:250], axis=0))

window_size=50
NN_tructure=(16,16)
init=[[] for __ in range(window_size)]
record=[]
record_2=[]

for i in range(N_train, N_total):
    print(i)
    
    test_value=0
    test_value_2=0
    for w in range( window_size//2, window_size ):
        if not init[w]:
            init_from=None
        else:
            init_from=init[w]
        model = NumpyLogisticNN(input_dim=dim, hidden=NN_tructure, init_from= init_from , l2=1e-4 )
        model.fit(
            data[(i-2*w):(i-w)], data[(i-w):i],
            epochs=30, batch_size=i//10, lr=2e-3,
            X_test=data[0:N_train],         # enables drift-based early stop
            es_rel=0.01 ,         
            es_min_epochs=2,       # wait a couple epochs before checking
            es_norm_ord=2  # L2 norm over f(X_test)
        )
        init[w]=model
            
        test_value=max(test_value, np.mean(model.f_theta(data[0:N_train])))
        test_value_2=max(test_value_2, np.linalg.norm(np.mean(data[0:(i-w)], axis=0) - np.mean(data[(i-w):i], axis=0))
 )
    record.append(test_value)
    record_2.append(test_value_2)
#W0, B0=init[26].get_params()    
#record = np.array(record)
plt.plot(record )             # x = 0,1,2,3; line plot
plt.xlabel("index"); 
plt.ylabel("value"); 
plt.grid(True)
plt.show()


"""


######################
######################



import numpy as np
import csv
import matplotlib.pyplot as plt
path = "/Users/daren/git/NN_change/bill.csv"

# Discover number of columns
with open(path, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
ncols = len(header)

# --- 1) Load the first column (dates) as strings ---
dates = np.genfromtxt(
    path,
    delimiter=",",
    skip_header=1,
    usecols=(0,),
    dtype="U128"     # unicode string
)

# (Optional) If your dates are ISO-like (YYYY-MM-DD), convert to datetime64:
# try:
#     dates = dates.astype("datetime64[ns]")
# except ValueError:
#     pass  # keep as strings if parsing fails

# --- 2) Load the numeric columns (skip the first column) ---
usecols_num = tuple(range(1, ncols))
A = np.genfromtxt(
    path,
    delimiter=",",
    skip_header=1,
    usecols=usecols_num,
    dtype=float,
    missing_values=["", "NA", "NaN", "nan", "."],
    filling_values=np.nan,
)

# --- 3) Drop rows with any NaN in A and keep dates aligned ---
mask = ~np.isnan(A).any(axis=1)
A = A[mask]
dates = dates[mask]
dates=dates[::-1]

data=A[::-1,]
data1=data[100:175]
data2=data[175:225]
#np.linalg.norm(np.mean(data1, axis=0) -np.mean(data2, axis=0))
data[100: 225]



# ---- run PCA and get the N×2 projection ----
X = data[100: 225]

# 2) SVD on centered data (A = U S V^T). Rows = observations.
#    The right-singular vectors (rows of V^T) are principal axes.
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# 3) Take first two principal components (columns)
PCs = Vt[:2].T                  # shape: (dim, 2)

# 4) Project to 2D
Z = X @ PCs                     # shape: (N, 2)


Z1 = data1@ PCs   
Z2 = data2@ PCs   
# ---- heat map of the 2D projection (density) ----


def kde2d_numpy(B, gridsize=200, extent=None, bw=None, rule="scott"):
    B = np.asarray(B)
    N, d = B.shape
    assert d == 2, "B must be (N,2)"

    # Plot window
    if extent is None:
        p1 = np.percentile(B, 1, axis=0)
        p99 = np.percentile(B, 99, axis=0)
        pad = 0.05 * (p99 - p1 + 1e-12)
        extent = [p1[0]-pad[0], p99[0]+pad[0], p1[1]-pad[1], p99[1]+pad[1]]

    x = np.linspace(extent[0], extent[1], gridsize)
    y = np.linspace(extent[2], extent[3], gridsize)
    X, Y = np.meshgrid(x, y)
    G = np.column_stack([X.ravel(), Y.ravel()])  # (M,2)

    # Whiten using covariance
    mu = B.mean(axis=0)
    C = np.cov(B.T) + 1e-9 * np.eye(2)
    L = np.linalg.cholesky(C)
    invL = np.linalg.inv(L)
    Bw = (B - mu) @ invL.T
    Gw = (G - mu) @ invL.T

    # Bandwidth in whitened space
    if bw is None:
        if rule.lower() == "silverman":
            bw = (N * (2 + 2) / 4.0) ** (-1.0 / (2 + 4))
        else:  # Scott
            bw = N ** (-1.0 / (2 + 4))
    h2 = bw ** 2

    # Gaussian kernel sum in whitened space
    Gw2 = np.sum(Gw**2, axis=1, keepdims=True)     # (M,1)
    Bw2 = np.sum(Bw**2, axis=1, keepdims=True).T   # (1,N)
    cross = Gw @ Bw.T                               # (M,N)
    dist2 = (Gw2 + Bw2 - 2.0 * cross) / h2

    norm_const = 1.0 / (2.0 * np.pi)               # for d=2
    detL = np.linalg.det(L)
    D = (norm_const / (N * (bw**2) * detL)) * np.exp(-0.5 * dist2).sum(axis=1)
    D = D.reshape(gridsize, gridsize)

    return X, Y, D, extent

def plot_kde2d_numpy(B, gridsize=200, extent=None, bw=None, rule="scott"):
    X, Y, D, extent = kde2d_numpy(B, gridsize, extent, bw, rule)
    plt.figure(figsize=(6, 5))
    plt.imshow(D, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="density")
    plt.xlabel("First pc")
    plt.ylabel("Second pc")
    plt.title("2D KDE of B (NumPy)")
    # Optional contours:
    # plt.contour(X, Y, D, levels=10, linewidths=0.8)
    plt.tight_layout()
    plt.show()

 


from matplotlib.ticker import FormatStrFormatter

# assumes kde2d_numpy(B, gridsize, extent, bw, rule) is defined

def _robust_extent(B):
    p1  = np.percentile(B, 1, axis=0)
    p99 = np.percentile(B, 99, axis=0)
    pad = 0.05 * (p99 - p1 + 1e-12)
    return [p1[0] - pad[0], p99[0] + pad[0], p1[1] - pad[1], p99[1] + pad[1]]

def _make_ticks(vmin, vmax, count):
    if not np.isfinite(vmin) or not np.isfinite(vmax): return []
    if np.isclose(vmin, vmax): return [vmin] * count
    return np.linspace(vmin, vmax, count)

def _apply_tick_format(ax, digits=1):
    fmt = f"%.{digits}f"
    ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

def plot_kde2d_numpy_two_adaptive(
    B1, B2,
    gridsize=200,
    bw=None,
    rule="scott",
titles=("Joint distribution (projected on the first two PCs) \nBEFORE the change",
            "Joint distribution (projected on the first two PCs)  \nAFTER the change"),
    share_colorbar=False,
    tick_count=5,
    tick_digits=1,
    title_size=15,          # <— new
    title_pad=10,            # <— new (pixels above axes)
    title_linespacing=1.1,  # <— new (line spacing for multi-line titles)
):
    B1 = np.asarray(B1); B2 = np.asarray(B2)
    assert B1.ndim == 2 and B1.shape[1] == 2
    assert B2.ndim == 2 and B2.shape[1] == 2

    extent1 = _robust_extent(B1)
    extent2 = _robust_extent(B2)

    X1, Y1, D1, extent1 = kde2d_numpy(B1, gridsize=gridsize, extent=extent1, bw=bw, rule=rule)
    X2, Y2, D2, extent2 = kde2d_numpy(B2, gridsize=gridsize, extent=extent2, bw=bw, rule=rule)

    vmin = vmax = None
    if share_colorbar:
        vmin, vmax = 0.0, max(D1.max(), D2.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im1 = axes[0].imshow(D1, origin="lower", extent=extent1, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title(
    titles[0],
    fontsize=title_size,      # <- size
    pad=title_pad,            # <- distance above axes (points)
    linespacing=title_linespacing)    
    axes[0].set_xlabel("First PC",fontsize=12); axes[0].set_ylabel("Second PC",fontsize=12)

    im2 = axes[1].imshow(D2, origin="lower", extent=extent2, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title(
    titles[1],
    fontsize=title_size,
    pad=title_pad,
    linespacing=title_linespacing)
    axes[1].set_xlabel("First PC",fontsize=12); axes[1].set_ylabel("Second PC",fontsize=12)

    # same number of ticks per axis (per subplot)
    xt1 = _make_ticks(extent1[0], extent1[1], tick_count)
    yt1 = _make_ticks(extent1[2], extent1[3], tick_count)
    axes[0].set_xticks(xt1); axes[0].set_yticks(yt1)

    xt2 = _make_ticks(extent2[0], extent2[1], tick_count)
    yt2 = _make_ticks(extent2[2], extent2[3], tick_count)
    axes[1].set_xticks(xt2); axes[1].set_yticks(yt2)

    # format tick labels to one decimal place
    _apply_tick_format(axes[0], digits=tick_digits)
    _apply_tick_format(axes[1], digits=tick_digits)

    if share_colorbar:
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist()) 
    else:
        fig.colorbar(im1, ax=axes[0]) 
        fig.colorbar(im2, ax=axes[1]) 

    plt.show()

# Example:
# plot_kde2d_numpy_two_adaptive(B1, B2, tick_count=5, tick_digits=1)


# Example:
# plot_kde2d_numpy_two_adaptive(B1, B2, gridsize=220, rule="scott", tick_count=5)

# Example:
# plot_kde2d_numpy_two_adaptive(B1, B2, gridsize=220, rule="scott", tick_count=5)


# Example:
# plot_kde2d_numpy_two_adaptive(B1, B2, gridsize=220, rule="scott", share_colorbar=False)

# Example:
# plot_kde2d_numpy_two(B1, B2, gridsize=220, rule="scott", share_colorbar=True)

plot_kde2d_numpy_two_adaptive(Z1, Z2)
#plot_kde2d_numpy(Z1 )
#plot_kde2d_numpy(Z2 )
# Usage:
# plot_kde2d_numpy(B, gridsize=220, rule="scott")
# plot_kde2d_numpy(B, gridsize=160, bw=1.2)  # custom smoothing


