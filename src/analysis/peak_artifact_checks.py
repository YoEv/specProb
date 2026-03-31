import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform


EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
RESULTS_DIR = "results/peak_artifact_investigation"
N_COEFFS = 33  # rfft(64)


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def describe_band_layout(
    n_coeffs: int = N_COEFFS,
    band_width: int = 4,
    out_filename: str = "band_layout_fft.txt",
) -> None:
    """Describe how we split coefficients into bands, and check for gaps / overlaps.

    This对应 plan 里“检查 band 拼接与覆盖是否合理，并文档化 band 划分方式”。
    它不直接解析脚本，而是用和脚本相同的逻辑构造 bands，然后检查：
    - 每个 band 的起止 index；
    - 所有 band 的并集是否正好覆盖 [0, n_coeffs) 且没有重复。
    """
    _ensure_results_dir()

    num_bands = n_coeffs // band_width
    bands = {b: list(range(b * band_width, (b + 1) * band_width)) for b in range(num_bands)}

    covered = sorted({idx for _, idxs in bands.items() for idx in idxs})
    expected = list(range(num_bands * band_width))

    gaps = sorted(set(expected) - set(covered))
    duplicates = sorted([idx for idx in covered if covered.count(idx) > 1])

    # Note: 对于 33 个 coeff，我们现在只用前 32 个 (8 bands × 4)，最后一个系数目前没用到。
    unused = sorted(set(range(n_coeffs)) - set(covered))

    lines = []
    lines.append(f"n_coeffs = {n_coeffs}, band_width = {band_width}, num_bands = {num_bands}")
    lines.append("")
    lines.append("Bands (0-based indices, inclusive ranges):")
    for b, idxs in bands.items():
        lines.append(f"  Band {b}: coeffs {idxs[0]}–{idxs[-1]} (len={len(idxs)})")
    lines.append("")
    lines.append(f"Covered indices (min..max): {covered[0]}..{covered[-1]}")
    lines.append(f"Gaps within [0, {num_bands * band_width - 1}]: {gaps or 'None'}")
    lines.append(f"Duplicates within used bins: {sorted(set(duplicates)) or 'None'}")
    lines.append(f"Unused coeff indices in [0, {n_coeffs-1}]: {unused or 'None'}")

    out_path = os.path.join(RESULTS_DIR, out_filename)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[peak_artifact_checks] Wrote band layout description to {out_path}")


def run_single_sample_fft_check(
    embeddings_path: str = EMBEDDINGS_PATH,
    sample_idx: int = 0,
    dim_idx: int = 0,
    window_type: Optional[str] = None,
) -> None:
    """Numerically compare np.fft.rfft with our apply_transform for one (sample, dim).

    Writes a small text report to
    results/peak_artifact_investigation/single_sample_fft_check.txt.
    """
    _ensure_results_dir()
    X, _ = load_data(embeddings_path)

    if sample_idx < 0 or sample_idx >= X.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} out of range (0, {X.shape[0]-1})")
    if dim_idx < 0 or dim_idx >= X.shape[1]:
        raise IndexError(f"dim_idx {dim_idx} out of range (0, {X.shape[1]-1})")

    x = X[sample_idx, dim_idx, :]  # (64,)
    manual = np.abs(np.fft.rfft(x, norm="ortho"))

    X_slice = X[sample_idx : sample_idx + 1, dim_idx : dim_idx + 1, :]
    coeffs = apply_transform(X_slice, transform_type="fft", axis=2, window_type=window_type)
    pipe = coeffs[0, 0, :]

    diff = manual - pipe
    max_abs_diff = float(np.max(np.abs(diff)))
    l2 = float(np.linalg.norm(diff))

    out_path = os.path.join(RESULTS_DIR, "single_sample_fft_check.txt")
    with open(out_path, "w") as f:
        f.write(f"embeddings_path: {embeddings_path}\n")
        f.write(f"sample_idx: {sample_idx}, dim_idx: {dim_idx}\n")
        f.write(f"window_type: {window_type}\n\n")
        f.write("manual_fft (abs rfft, norm='ortho'):\n")
        f.write(" ".join(f"{v:.6f}" for v in manual) + "\n\n")
        f.write("pipeline_fft (apply_transform -> coeffs):\n")
        f.write(" ".join(f"{v:.6f}" for v in pipe) + "\n\n")
        f.write("diff (manual - pipeline):\n")
        f.write(" ".join(f"{v:.6e}" for v in diff) + "\n\n")
        f.write(f"max_abs_diff: {max_abs_diff:.6e}\n")
        f.write(f"l2_norm_diff: {l2:.6e}\n")

    print(f"[peak_artifact_checks] Wrote single-sample FFT comparison to {out_path}")


def run_random_vector_fft_pipeline(
    n_trials: int = 5,
    shape: Sequence[int] = (1, 1536, 64),
    window_type: Optional[str] = None,
) -> None:
    """Run FFT pipeline on random Gaussian inputs and visualise mean spectrum.

    If random inputs也产生类似 3-peaks，就更像是数值 / 实现层面的 artifact。
    """
    _ensure_results_dir()
    all_mean_specs = []

    for t in range(n_trials):
        X_rand = np.random.randn(*shape).astype(np.float32)
        coeffs = apply_transform(X_rand, transform_type="fft", axis=2, window_type=window_type)
        mean_spec = np.mean(coeffs, axis=(0, 1))
        all_mean_specs.append(mean_spec)

    all_mean_specs = np.stack(all_mean_specs, axis=0)  # (n_trials, N_COEFFS)
    x = np.arange(all_mean_specs.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_trials):
        ax.plot(x, all_mean_specs[i], alpha=0.3, color="C0")
    ax.plot(x, np.mean(all_mean_specs, axis=0), color="black", linewidth=2, label="mean over trials")
    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Mean magnitude")
    ax.set_title(f"Random Gaussian FFT spectra (n_trials={n_trials}, window={window_type})")
    ax.legend()
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, f"random_fft_spectrum_window={window_type or 'none'}.png")
    npy_path = os.path.join(RESULTS_DIR, f"random_fft_spectrum_window={window_type or 'none'}.npy")
    plt.savefig(png_path)
    plt.close(fig)
    np.save(npy_path, all_mean_specs)

    print(f"[peak_artifact_checks] Saved random FFT spectra to {png_path} and {npy_path}")


def plot_per_sample_spectra_for_genre(
    target_genre: str,
    max_samples: int = 200,
    window_type: Optional[str] = None,
) -> None:
    """Plot per-sample spectra for one genre (thin transparent lines + mean).

    用来检查 3 个 peak 是大多数样本都有，还是少数 outlier 导致的平均效应。
    """
    _ensure_results_dir()
    X, y_str = load_data(EMBEDDINGS_PATH)
    mask = y_str == target_genre
    if not np.any(mask):
        raise ValueError(f"No samples for genre '{target_genre}'")

    X_genre = X[mask]
    n_genre = X_genre.shape[0]
    if n_genre > max_samples:
        X_genre = X_genre[:max_samples]
        n_used = max_samples
    else:
        n_used = n_genre

    coeffs = apply_transform(X_genre, transform_type="fft", axis=2, window_type=window_type)
    # (n_used, F, N_COEFFS) -> per-sample spectrum (mean over features)
    per_sample = np.mean(coeffs, axis=1)  # (n_used, N_COEFFS)
    mean_spec = np.mean(per_sample, axis=0)

    x = np.arange(per_sample.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_used):
        ax.plot(x, per_sample[i], color="C0", alpha=0.1)
    ax.plot(x, mean_spec, color="black", linewidth=2, label="mean over samples")
    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Mean magnitude across features")
    ax.set_xticks([0, per_sample.shape[1] // 2, per_sample.shape[1] - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_title(
        f"Per-sample FFT spectra for genre={target_genre} (n_used={n_used}, window={window_type})"
    )
    ax.legend()
    plt.tight_layout()

    suffix = target_genre.replace(" ", "_").replace("/", "_")
    png_path = os.path.join(
        RESULTS_DIR, f"per_sample_spectrum_{suffix}_window={window_type or 'none'}.png"
    )
    plt.savefig(png_path)
    plt.close(fig)

    print(f"[peak_artifact_checks] Saved per-sample spectra plot for {target_genre} to {png_path}")


def run_time_permutation_experiment(
    max_samples: int = 500,
    random_state: int = 42,
    window_type: Optional[str] = None,
) -> None:
    """
    Permute the time axis of real embeddings and compare spectra (mean magnitude).

    - Sample up to max_samples from the full dataset (to keep runtime manageable).
    - For each sample, apply the SAME random permutation to all feature dims.
    - Compute mean magnitude spectrum before and after permutation:
        mean_orig[k] = mean over (sample, feature)
        mean_perm[k] = same after time permutation
    - If the 3 peaks strongly depend on original ordering, they should weaken/disappear
      under permutation; if they remain, the structure is more content/statistics-driven.
    """
    _ensure_results_dir()
    rng = np.random.RandomState(random_state)

    X, _ = load_data(EMBEDDINGS_PATH)  # (B, 1536, 64)
    B = X.shape[0]
    if B > max_samples:
        idx = rng.choice(B, size=max_samples, replace=False)
        X = X[idx]
        B_used = max_samples
    else:
        B_used = B

    # Original order spectra
    coeffs_orig = apply_transform(X, transform_type="fft", axis=2, window_type=window_type)
    mean_orig = np.mean(coeffs_orig, axis=(0, 1))  # (N_COEFFS,)

    # Time-permuted spectra
    X_perm = X.copy()
    T = X.shape[2]
    perm = rng.permutation(T)
    X_perm = X_perm[:, :, perm]

    coeffs_perm = apply_transform(X_perm, transform_type="fft", axis=2, window_type=window_type)
    mean_perm = np.mean(coeffs_perm, axis=(0, 1))

    x = np.arange(mean_orig.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_orig, label=f"Original (n={B_used})", color="black", linewidth=2)
    ax.plot(x, mean_perm, label=f"Time-permuted (same set)", color="C1", linewidth=2, linestyle="--")
    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Mean magnitude across samples & features")
    ax.set_xticks([0, mean_orig.shape[0] // 2, mean_orig.shape[0] - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_title(
        f"Effect of time permutation on mean FFT spectrum\n"
        f"(max_samples={B_used}, window={window_type or 'none'})"
    )
    ax.legend()
    plt.tight_layout()

    png_path = os.path.join(
        RESULTS_DIR, f"permuted_time_spectrum_window={window_type or 'none'}.png"
    )
    plt.savefig(png_path)
    plt.close(fig)

    print(
        f"[peak_artifact_checks] Saved time permutation spectrum comparison to {png_path} "
        f"(B_used={B_used})"
    )


def run_layout_variants_experiment(
    max_samples: int = 500,
    random_state: int = 42,
    window_type: Optional[str] = None,
) -> None:
    """
    Compare different flatten/FFT layouts to see if peak positions follow layout.

    Variants:
        1) Time-major baseline (current pipeline):
           - FFT along time axis of X ∈ R^{B×F×T}, mean over (sample, feature).
        2) Time-only (freq-aggregated) FFT:
           - First average over feature dims to get X_time ∈ R^{B×T}, then FFT over T.

    （更复杂的 freq-major 等可以在需要时追加，这里先验证“只取 time axis”是否仍有三峰。）
    """
    _ensure_results_dir()
    rng = np.random.RandomState(random_state)

    X, _ = load_data(EMBEDDINGS_PATH)  # (B, 1536, 64)
    B = X.shape[0]
    if B > max_samples:
        idx = rng.choice(B, size=max_samples, replace=False)
        X = X[idx]
        B_used = max_samples
    else:
        B_used = B

    # 1) Baseline: current pipeline (FFT over time on full (B, F, T))
    coeffs_full = apply_transform(X, transform_type="fft", axis=2, window_type=window_type)
    mean_full = np.mean(coeffs_full, axis=(0, 1))  # (N_COEFFS,)

    # 2) Time-only: average over feature dims, then FFT over time
    X_time = np.mean(X, axis=1)  # (B, T)
    coeffs_time = np.abs(
        np.fft.rfft(X_time, axis=1, norm="ortho")
    )  # (B, N_COEFFS) – same N_COEFFS (T=64 → 33)
    mean_time = np.mean(coeffs_time, axis=0)  # (N_COEFFS,)

    x = np.arange(mean_full.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_full, label="Baseline: FFT on full (F×T)", color="black", linewidth=2)
    ax.plot(x, mean_time, label="Time-only: mean over F then FFT", color="C2", linewidth=2, linestyle="--")
    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Mean magnitude")
    ax.set_xticks([0, mean_full.shape[0] // 2, mean_full.shape[0] - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_title(
        f"Layout variants: full vs time-only FFT\n"
        f"(max_samples={B_used}, window={window_type or 'none'})"
    )
    ax.legend()
    plt.tight_layout()

    png_path = os.path.join(
        RESULTS_DIR, f"layout_variants_spectrum_window={window_type or 'none'}.png"
    )
    plt.savefig(png_path)
    plt.close(fig)

    print(
        f"[peak_artifact_checks] Saved layout variants spectrum comparison to {png_path} "
        f"(B_used={B_used})"
    )


