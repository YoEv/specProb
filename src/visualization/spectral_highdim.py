import os
from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # type: ignore


def compute_dimension_spectrum_matrix(
    X_freq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-dimension mean/std spectra.

    Args:
        X_freq: array of shape (n_samples, n_dims, n_coeffs)

    Returns:
        M_mean: (n_dims, n_coeffs)
        M_std:  (n_dims, n_coeffs)
    """
    assert X_freq.ndim == 3, f"Expected 3D array, got {X_freq.shape}"
    M_mean = np.mean(X_freq, axis=0)
    M_std = np.std(X_freq, axis=0)
    return M_mean, M_std


def plot_heatmap_dim_x_freq(
    M: np.ndarray,
    out_path: str,
    title: str,
    normalize_per_dim: bool = True,
    sort_by_energy: bool = True,
) -> None:
    """
    Plot a heatmap of dimension × frequency spectra.

    Args:
        M: (n_dims, n_coeffs) mean spectra
        out_path: where to save the PNG
        title: figure title
        normalize_per_dim: whether to L2-normalize each row
        sort_by_energy: whether to sort rows by total energy (L2 norm)
    """
    M_plot = M.copy()
    if normalize_per_dim:
        norms = np.linalg.norm(M_plot, axis=1, keepdims=True) + 1e-8
        M_plot = M_plot / norms

    if sort_by_energy:
        energy = np.linalg.norm(M_plot, axis=1)
        order = np.argsort(-energy)
        M_plot = M_plot[order]

    n_dims, n_coeffs = M_plot.shape
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(M_plot, aspect="auto", origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Embedding dimension (sorted)")
    ax.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def cluster_dimensions_kmeans(
    M: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster dimensions based on their spectral shape.

    Args:
        M: (n_dims, n_coeffs)

    Returns:
        labels: (n_dims,) cluster assignment for each dimension.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(M)
    return labels


def plot_cluster_mean_spectra(
    M: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """
    Plot mean spectrum for each cluster.

    Args:
        M: (n_dims, n_coeffs)
        labels: (n_dims,)
    """
    n_clusters = int(labels.max()) + 1
    n_coeffs = M.shape[1]
    x = np.arange(n_coeffs)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        mean_k = np.mean(M[mask], axis=0)
        ax.plot(x, mean_k, label=f"cluster {k} (n={np.sum(mask)})")

    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Mean magnitude (normalized row-wise)")
    ax.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_title(title)
    ax.legend(fontsize="small")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_per_sample_curves_for_dims(
    X_freq: np.ndarray,
    dim_indices: Sequence[int],
    genre_mask: np.ndarray,
    out_path: str,
    title: str,
    max_samples: int = 200,
) -> None:
    """
    For a small set of dimensions, plot per-sample spectra (thin) + mean (bold).

    Args:
        X_freq: (n_samples, n_dims, n_coeffs)
        dim_indices: list of dims to visualise
        genre_mask: boolean mask of shape (n_samples,) selecting samples (e.g. a genre)
        out_path: where to save PNG
    """
    assert X_freq.ndim == 3
    n_samples, n_dims, n_coeffs = X_freq.shape
    dim_indices = [d for d in dim_indices if 0 <= d < n_dims]

    mask = genre_mask.astype(bool)
    X_sel = X_freq[mask]
    n_used = min(max_samples, X_sel.shape[0])
    X_sel = X_sel[:n_used]

    x = np.arange(n_coeffs)
    n_plots = len(dim_indices)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(
        1, n_plots, figsize=(4 * n_plots, 3), sharey=True
    )
    if n_plots == 1:
        axes = [axes]

    for ax, d in zip(axes, dim_indices):
        per_sample = X_sel[:, d, :]  # (n_used, n_coeffs)
        mean_spec = np.mean(per_sample, axis=0)
        for i in range(n_used):
            ax.plot(x, per_sample[i], color="C0", alpha=0.1)
        ax.plot(x, mean_spec, color="black", linewidth=2)
        ax.set_title(f"dim {d} (n={n_used})")
        ax.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
        ax.set_xticklabels(["L", "M", "H"])
        ax.set_xlabel("Frequency coefficient")
    axes[0].set_ylabel("Magnitude")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_3d_sample_freq_surface(
    X_freq: np.ndarray,
    out_path: str,
    title: str,
    max_samples: int = 200,
) -> None:
    """
    Plot a 3D surface of (sample index, frequency coefficient, mean magnitude across dims).

    This对应“不给 sample 聚合成一个 mean 曲线，而是把 sample 当作一个轴”。

    Args:
        X_freq: (n_samples, n_dims, n_coeffs)
    """
    assert X_freq.ndim == 3
    n_samples, n_dims, n_coeffs = X_freq.shape

    # Mean over dims but保留 sample 轴
    mean_over_dims = np.mean(X_freq, axis=1)  # (n_samples, n_coeffs)
    n_used = min(max_samples, n_samples)
    Z = mean_over_dims[:n_used]  # (n_used, n_coeffs)

    S, K = np.meshgrid(
        np.arange(n_used),
        np.arange(n_coeffs),
        indexing="ij",
    )  # S: sample index, K: coeff index

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        K, S, Z, cmap="viridis", linewidth=0, antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Sample index")
    ax.set_zlabel("Mean magnitude (over dims)")
    ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_3d_dim_freq_surface(
    S: np.ndarray,
    out_path: str,
    title: str,
    dim_step: int = 4,
) -> None:
    """Plot a 3D surface of (dimension index, frequency coefficient, magnitude) for one sample.

    Args:
        S: (n_dims, n_coeffs) single-sample spectrum over all dims.
    """
    assert S.ndim == 2
    n_dims, n_coeffs = S.shape

    # Subsample dim axis to avoid overly dense mesh, but keep overall structure
    dims = np.arange(0, n_dims, dim_step)
    Z = S[dims, :]  # (n_sub_dims, n_coeffs)
    D, K = np.meshgrid(dims, np.arange(n_coeffs), indexing="ij")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        K, D, Z, cmap="viridis", linewidth=0, antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Dimension index (subsampled)")
    ax.set_zlabel("Magnitude")
    ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

