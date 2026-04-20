"""
Band-pair / band-triplet multiclass probing on (B, F, T) embeddings.

Given an embeddings NPZ (same format as `run_generic_fft_band_probe.py`), applies
FFT/DCT along the time axis, splits the `n_coeffs` axis into `K` contiguous bands
of width `band_width`, then trains *GPU, multinomial* linear probes for:

- solo per-band features  (K probes, sanity baseline for synergy/redundancy);
- cumulative prefix probes (K probes, for `prefix_delta` band ranking);
- all unordered 1+1 band pairs `{i, j}` with `i < j`  (C(K, 2) probes);
- all 1+2 "anchor + adjacent 2-block" triples `(i, [j, j+1])`
  with `j != i` and `j+1 != i`  (approximately K * (K-2) probes).

Outputs (under `--results_dir`):
- `pair_1p1_matrix.json`   : K x K accuracy matrix with solo on diagonal,
                             plus synergy / redundancy matrices.
- `synergy_1p1_matrix.png` : heatmap of acc(i+j) - max(acc(i), acc(j)).
- `redundancy_1p1_matrix.png` : heatmap of acc(i) + acc(j) - acc(i+j).
- `pair_1p2_summary.json`  : list of (anchor, [j, j+1]) probes with delta metrics.
- `band_ranking.json`      : solo / prefix_delta rankings + Spearman rank-corr.

All probes share `--random_state` (default 42) with the single-run policy of the
current `run_generic_fft_band_probe.py --label_mode multiclass` pipeline.
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spectral import apply_transform  # noqa: E402
from scripts.run_generic_fft_band_probe import (  # noqa: E402
    _load_npz_generic,
    _resolve_gpu_device,
    _run_probe_gpu_multiclass,
    _run_probe_gpu_batched_same_dim_multiclass,
    _run_probe_gpu_batched_prefix_multiclass,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _chunked(items: list, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _run_same_dim_in_chunks_lazy(
    task_index_specs: list[tuple[int, ...]],
    band_feats: list[np.ndarray],
    y_int: np.ndarray,
    n_classes: int,
    random_state: int,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    chunk_size: int,
    label: str,
) -> list[float]:
    """Batched same-dim multiclass probes, building features per-chunk.

    `task_index_specs` is a list where each entry is a tuple of band indices
    whose per-band feature arrays (from `band_feats`) get `np.concatenate`'d
    along the feature axis to form that task's input. Only one `chunk_size`
    worth of materialised features is held in RAM at a time.
    """
    total = len(task_index_specs)
    accs: list[float] = []
    if total == 0:
        return accs
    chunks = list(_chunked(task_index_specs, chunk_size))
    feat_dim_example = sum(band_feats[i].shape[1] for i in task_index_specs[0])
    approx_mb = (
        chunk_size
        * band_feats[0].shape[0]
        * feat_dim_example
        * 4
        / 1e6
    )
    print(
        f"[run_band_pair_probe] {label}: {total} tasks in "
        f"{len(chunks)} chunk(s) of up to {chunk_size}; feat_dim={feat_dim_example} "
        f"(~{approx_mb:.0f} MB per chunk task-tensor)",
        flush=True,
    )
    for ci, spec_chunk in enumerate(chunks):
        chunk_feats = [
            np.concatenate([band_feats[i] for i in spec], axis=-1)
            for spec in spec_chunk
        ]
        print(
            f"[run_band_pair_probe] {label}: chunk {ci + 1}/{len(chunks)} "
            f"built ({len(chunk_feats)} tasks, feat_dim={chunk_feats[0].shape[1]})",
            flush=True,
        )
        chunk_accs, _, _ = _run_probe_gpu_batched_same_dim_multiclass(
            X_task_features=chunk_feats,
            y_labels=y_int,
            n_classes=n_classes,
            random_state=random_state,
            test_size=0.2,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        accs.extend(float(a) for a in chunk_accs)
        del chunk_feats, chunk_accs
        gc.collect()
        try:
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
    assert len(accs) == total, f"{label}: expected {total} accs, got {len(accs)}"
    return accs


def _rank_vector(values: Sequence[float]) -> list[int]:
    """Return 0-indexed ranks (0 = highest value), ties broken by index."""
    sorted_idx = sorted(
        range(len(values)),
        key=lambda i: (-float(values[i]), i),
    )
    ranks = [0] * len(values)
    for r, i in enumerate(sorted_idx):
        ranks[i] = r
    return ranks


def _spearman(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """Spearman's rank correlation on two equally-sized 0-indexed rank vectors."""
    n = len(rank_a)
    if n != len(rank_b) or n < 2:
        return float("nan")
    d2 = sum((int(rank_a[i]) - int(rank_b[i])) ** 2 for i in range(n))
    denom = n * (n * n - 1)
    if denom == 0:
        return float("nan")
    return 1.0 - 6.0 * d2 / denom


def _symmetric_heatmap(
    matrix: np.ndarray,
    out_path: str,
    title: str,
    cbar_label: str,
    K: int,
    center_zero: bool = False,
    annotate: bool = True,
) -> None:
    """Save a K x K heatmap with optional zero-centered diverging colormap."""
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * K, 0.8 + 0.55 * K))
    if center_zero:
        vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Band j")
    ax.set_ylabel("Band i")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"B{i}" for i in range(K)], fontsize=8)
    ax.set_yticklabels([f"B{i}" for i in range(K)], fontsize=8)
    if annotate and K <= 16:
        for i in range(K):
            for j in range(K):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if (center_zero and abs(v) > 0.5 * vmax) else "black",
                    fontsize=7,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GPU multiclass band-pair / band-triplet probing on (B, F, T) "
            "embeddings NPZ. Produces pair/redundancy matrices and band rankings."
        )
    )
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument(
        "--label_key",
        type=str,
        default="genres",
        help="Name of the label array inside the NPZ (e.g. 'genres').",
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        default="fft",
        choices=["fft", "dct"],
    )
    parser.add_argument("--band_width", type=int, default=4)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/3rd_exp/band_pair_probe",
    )
    parser.add_argument("--prefix", type=str, default="generic")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["gpu", "rocm", "auto"],
        help="Multiclass probe is GPU-only; CPU backend is not supported here.",
    )
    parser.add_argument("--gpu_device", type=str, default="cuda")
    parser.add_argument("--gpu_epochs", type=int, default=400)
    parser.add_argument("--gpu_lr", type=float, default=0.05)
    parser.add_argument("--gpu_weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--pair_chunk_size",
        type=int,
        default=32,
        help=(
            "How many pair/triplet tasks to batch into a single GPU call. "
            "Lower this if you hit CUDA OOM."
        ),
    )
    parser.add_argument(
        "--max_pairs_1p2",
        type=int,
        default=-1,
        help=(
            "Optional cap on the number of 1+2 triplet probes (random subsample). "
            "-1 means keep them all."
        ),
    )
    parser.add_argument(
        "--skip_triplets",
        action="store_true",
        help="If set, skip the 1+2 triplet phase (solo + pair only).",
    )
    parser.add_argument(
        "--gpu_epochs_triplet",
        type=int,
        default=-1,
        help=(
            "Epochs for the 1+2 triplet phase. -1 means reuse --gpu_epochs. "
            "Lower this (e.g. 200) for faster triplet runs."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("[run_band_pair_probe] starting; resolving GPU device ...", flush=True)
    device = _resolve_gpu_device(args)
    print(
        f"[run_band_pair_probe] device={device} backend={args.backend} "
        f"random_state={args.random_state}",
        flush=True,
    )

    # ---- Load + encode labels ----
    print(
        f"[run_band_pair_probe] Loading NPZ: {args.npz_path} "
        f"(label_key={args.label_key})",
        flush=True,
    )
    X, y_str = _load_npz_generic(args.npz_path, args.label_key)
    print(
        f"[run_band_pair_probe] NPZ loaded. X.shape={X.shape} "
        f"X.dtype={X.dtype} y_str.shape={y_str.shape}",
        flush=True,
    )
    le = LabelEncoder()
    y_int = le.fit_transform(y_str).astype(np.int64)
    class_names: list[str] = [str(c) for c in le.classes_.tolist()]
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError(
            f"Band-pair probing requires >=2 classes; got {n_classes} in '{args.label_key}'."
        )
    counts = np.bincount(y_int, minlength=n_classes).tolist()
    n_total = int(y_int.shape[0])
    chance_uniform = 1.0 / float(n_classes)
    max_class_prop = float(max(counts)) / float(n_total) if n_total > 0 else 0.0
    print(
        f"[run_band_pair_probe] n_classes={n_classes} n_total={n_total} "
        f"chance_uniform={chance_uniform:.4f} max_class_prop={max_class_prop:.4f}"
    )

    # ---- Spectral transform ----
    print(
        f"[run_band_pair_probe] Applying {args.transform_type} on axis=2 ...",
        flush=True,
    )
    coeffs = apply_transform(X, transform_type=args.transform_type, axis=2)
    del X
    import gc
    gc.collect()
    coeff_bytes = coeffs.nbytes
    print(
        f"[run_band_pair_probe] coeffs ready. shape={coeffs.shape} "
        f"dtype={coeffs.dtype} size={coeff_bytes / 1e6:.1f} MB",
        flush=True,
    )
    B = coeffs.shape[0]
    n_features = coeffs.shape[1]
    n_coeffs = coeffs.shape[2]
    K = n_coeffs // args.band_width
    if K < 2:
        raise ValueError(
            f"Need at least 2 bands for pair analysis; band_width={args.band_width} "
            f"and n_coeffs={n_coeffs} gives K={K}."
        )
    print(
        f"[run_band_pair_probe] coeffs.shape={coeffs.shape} "
        f"(n_features={n_features}, n_coeffs={n_coeffs}, K={K}, "
        f"band_width={args.band_width}). "
        f"1+1 pair count = {K * (K - 1) // 2}; "
        f"1+2 triplet count (upper bound) = {K * (K - 2)}",
        flush=True,
    )

    # Per-band flattened features: each is (B, n_features * band_width).
    band_feats: list[np.ndarray] = [
        coeffs[:, :, i * args.band_width : (i + 1) * args.band_width].reshape(B, -1)
        for i in range(K)
    ]

    # ---- 1) Solo per-band probes (batched) ----
    solo_specs: list[tuple[int, ...]] = [(i,) for i in range(K)]
    solo_accs = _run_same_dim_in_chunks_lazy(
        task_index_specs=solo_specs,
        band_feats=band_feats,
        y_int=y_int,
        n_classes=n_classes,
        random_state=args.random_state,
        device=device,
        epochs=args.gpu_epochs,
        lr=args.gpu_lr,
        weight_decay=args.gpu_weight_decay,
        chunk_size=args.pair_chunk_size,
        label="solo_bands",
    )

    # ---- 2) Cumulative prefix probes (for prefix_delta) ----
    X_full_flat = coeffs.reshape(B, -1)
    prefix_lengths = [n_features * args.band_width * (i + 1) for i in range(K)]
    try:
        cum_accs_list, _, _ = _run_probe_gpu_batched_prefix_multiclass(
            X_full_features=X_full_flat,
            prefix_lengths=prefix_lengths,
            y_labels=y_int,
            n_classes=n_classes,
            random_state=args.random_state,
            test_size=0.2,
            device=device,
            epochs=args.gpu_epochs,
            lr=args.gpu_lr,
            weight_decay=args.gpu_weight_decay,
        )
        cum_accs = [float(a) for a in cum_accs_list]
    except RuntimeError as exc:
        print(
            "[run_band_pair_probe] batched prefix multiclass failed; "
            f"falling back to sequential. Reason: {exc}"
        )
        cum_accs = []
        cum_feat: np.ndarray | None = None
        for i in range(K):
            cum_feat = band_feats[i] if cum_feat is None else np.concatenate(
                [cum_feat, band_feats[i]], axis=-1
            )
            acc, _, _ = _run_probe_gpu_multiclass(
                X_features=cum_feat,
                y_labels=y_int,
                n_classes=n_classes,
                random_state=args.random_state,
                test_size=0.2,
                device=device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            cum_accs.append(float(acc))

    prefix_delta = [cum_accs[0]] + [
        cum_accs[i] - cum_accs[i - 1] for i in range(1, K)
    ]

    # ---- 3) 1+1 pair probes ----
    pairs_1p1: list[tuple[int, int]] = [
        (i, j) for i in range(K) for j in range(i + 1, K)
    ]
    print(f"[run_band_pair_probe] 1+1 pair count: {len(pairs_1p1)}", flush=True)
    pair_specs: list[tuple[int, ...]] = [(i, j) for (i, j) in pairs_1p1]
    pair_accs = _run_same_dim_in_chunks_lazy(
        task_index_specs=pair_specs,
        band_feats=band_feats,
        y_int=y_int,
        n_classes=n_classes,
        random_state=args.random_state,
        device=device,
        epochs=args.gpu_epochs,
        lr=args.gpu_lr,
        weight_decay=args.gpu_weight_decay,
        chunk_size=args.pair_chunk_size,
        label="pairs_1p1",
    )
    pair_acc_map: dict[tuple[int, int], float] = {
        (i, j): float(a) for (i, j), a in zip(pairs_1p1, pair_accs)
    }

    pair_matrix = np.full((K, K), np.nan, dtype=np.float64)
    synergy_matrix = np.zeros((K, K), dtype=np.float64)
    redundancy_matrix = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        pair_matrix[i, i] = float(solo_accs[i])
    for (i, j), a_ij in pair_acc_map.items():
        a_i = float(solo_accs[i])
        a_j = float(solo_accs[j])
        pair_matrix[i, j] = pair_matrix[j, i] = a_ij
        syn = a_ij - max(a_i, a_j)
        red = a_i + a_j - a_ij
        synergy_matrix[i, j] = synergy_matrix[j, i] = syn
        redundancy_matrix[i, j] = redundancy_matrix[j, i] = red

    # ---- 4a) Compute rankings (depend only on solo + cumulative) ----
    rank_by_solo = _rank_vector(solo_accs)
    rank_by_prefix_delta = _rank_vector(prefix_delta)
    spearman_rho = _spearman(rank_by_solo, rank_by_prefix_delta)
    order_by_solo_desc = sorted(range(K), key=lambda i: -float(solo_accs[i]))
    order_by_prefix_delta_desc = sorted(
        range(K), key=lambda i: -float(prefix_delta[i])
    )

    common_config = {
        "npz_path": args.npz_path,
        "label_key": args.label_key,
        "transform_type": args.transform_type,
        "band_width": int(args.band_width),
        "n_features": int(n_features),
        "n_coeffs": int(n_coeffs),
        "K": int(K),
        "random_state": int(args.random_state),
        "backend": args.backend,
        "gpu_device": str(device),
        "gpu_epochs": int(args.gpu_epochs),
        "gpu_lr": float(args.gpu_lr),
        "gpu_weight_decay": float(args.gpu_weight_decay),
        "pair_chunk_size": int(args.pair_chunk_size),
        "max_pairs_1p2": int(args.max_pairs_1p2),
        "skip_triplets": bool(args.skip_triplets),
        "gpu_epochs_triplet": int(
            args.gpu_epochs_triplet if args.gpu_epochs_triplet > 0 else args.gpu_epochs
        ),
        "results_dir": args.results_dir,
        "prefix": args.prefix,
        "class_names": class_names,
        "class_counts": {c: int(counts[i]) for i, c in enumerate(class_names)},
        "n_total": int(n_total),
        "chance": {
            "uniform": float(chance_uniform),
            "max_class_prop": float(max_class_prop),
        },
    }

    # ---- 4b) Save pair matrix + ranking + heatmaps IMMEDIATELY ----
    # These are the main deliverables; write them before the (long) triplet
    # phase so that a time-limit / OOM during triplets doesn't wipe them out.
    pair_matrix_path = os.path.join(
        args.results_dir, f"{args.prefix}_pair_1p1_matrix.json"
    )
    with open(pair_matrix_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pair_list": [[int(i), int(j)] for (i, j) in pairs_1p1],
                "pair_acc": [float(pair_acc_map[p]) for p in pairs_1p1],
                "solo_acc": [float(a) for a in solo_accs],
                "pair_matrix": pair_matrix.tolist(),
                "synergy_matrix": synergy_matrix.tolist(),
                "redundancy_matrix": redundancy_matrix.tolist(),
                "config": common_config,
            },
            f,
            indent=4,
        )
    print(
        f"[run_band_pair_probe] 1+1 pair matrix saved: {pair_matrix_path}",
        flush=True,
    )

    ranking_path = os.path.join(
        args.results_dir, f"{args.prefix}_band_ranking.json"
    )
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "K": int(K),
                "solo_acc": [float(a) for a in solo_accs],
                "cumulative_acc": [float(a) for a in cum_accs],
                "prefix_delta": [float(a) for a in prefix_delta],
                "rank_by_solo": rank_by_solo,
                "rank_by_prefix_delta": rank_by_prefix_delta,
                "order_by_solo_desc": order_by_solo_desc,
                "order_by_prefix_delta_desc": order_by_prefix_delta_desc,
                "spearman_rho_solo_vs_prefix_delta": (
                    float(spearman_rho) if not math.isnan(spearman_rho) else None
                ),
                "config": common_config,
            },
            f,
            indent=4,
        )
    print(
        f"[run_band_pair_probe] Band ranking saved: {ranking_path}",
        flush=True,
    )

    heatmap_pair_acc_path = os.path.join(
        args.results_dir, f"{args.prefix}_pair_1p1_acc_matrix.png"
    )
    _symmetric_heatmap(
        matrix=pair_matrix,
        out_path=heatmap_pair_acc_path,
        title=f"Pair acc (1+1) — {args.prefix}",
        cbar_label="overall acc",
        K=K,
        center_zero=False,
    )
    synergy_path = os.path.join(
        args.results_dir, f"{args.prefix}_synergy_1p1_matrix.png"
    )
    _symmetric_heatmap(
        matrix=synergy_matrix,
        out_path=synergy_path,
        title=f"Synergy acc(i+j) - max(acc(i), acc(j)) — {args.prefix}",
        cbar_label="synergy",
        K=K,
        center_zero=True,
    )
    redundancy_path = os.path.join(
        args.results_dir, f"{args.prefix}_redundancy_1p1_matrix.png"
    )
    _symmetric_heatmap(
        matrix=redundancy_matrix,
        out_path=redundancy_path,
        title=f"Redundancy acc(i) + acc(j) - acc(i+j) — {args.prefix}",
        cbar_label="redundancy",
        K=K,
        center_zero=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    x_axis = np.arange(K)
    ax.plot(x_axis, solo_accs, "o-", label="solo_acc")
    ax.plot(x_axis, cum_accs, "s--", label="cumulative_acc")
    ax.plot(x_axis, prefix_delta, "^:", label="prefix_delta")
    ax.axhline(
        y=chance_uniform, color="r", linestyle="--", linewidth=0.8,
        label=f"chance {chance_uniform:.2f}",
    )
    ax.set_xlabel("Band index (low → high freq)")
    ax.set_ylabel("overall acc")
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f"B{i}" for i in range(K)], fontsize=7, rotation=90)
    ax.set_title(f"Per-band ranking curves — {args.prefix}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    ranking_plot_path = os.path.join(
        args.results_dir, f"{args.prefix}_band_ranking.png"
    )
    fig.savefig(ranking_plot_path)
    plt.close(fig)
    print(
        f"[run_band_pair_probe] Pair/ranking figures saved under {args.results_dir}",
        flush=True,
    )

    # ---- 5) 1+2 triplet probes (optional, slow) ----
    if args.skip_triplets:
        print(
            "[run_band_pair_probe] --skip_triplets set; skipping 1+2 triplet phase.",
            flush=True,
        )
        _print_summary(
            solo_accs, cum_accs, prefix_delta,
            order_by_solo_desc, order_by_prefix_delta_desc, spearman_rho,
        )
        return

    triplets_1p2: list[tuple[int, int, int]] = []
    for i in range(K):
        for j in range(K - 1):
            k = j + 1
            if j == i or k == i:
                continue
            triplets_1p2.append((i, j, k))
    if args.max_pairs_1p2 > 0 and len(triplets_1p2) > args.max_pairs_1p2:
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(len(triplets_1p2), size=args.max_pairs_1p2, replace=False)
        triplets_1p2 = [triplets_1p2[int(t)] for t in sorted(idx)]
    print(
        f"[run_band_pair_probe] 1+2 triplet count: {len(triplets_1p2)}",
        flush=True,
    )

    triplet_epochs = (
        args.gpu_epochs_triplet if args.gpu_epochs_triplet > 0 else args.gpu_epochs
    )
    print(
        f"[run_band_pair_probe] triplet phase epochs={triplet_epochs} "
        f"(override -1 => reuse gpu_epochs={args.gpu_epochs})",
        flush=True,
    )

    if triplets_1p2:
        tri_specs: list[tuple[int, ...]] = [
            (i, j, k) for (i, j, k) in triplets_1p2
        ]
        tri_accs = _run_same_dim_in_chunks_lazy(
            task_index_specs=tri_specs,
            band_feats=band_feats,
            y_int=y_int,
            n_classes=n_classes,
            random_state=args.random_state,
            device=device,
            epochs=triplet_epochs,
            lr=args.gpu_lr,
            weight_decay=args.gpu_weight_decay,
            chunk_size=args.pair_chunk_size,
            label="triplets_1p2",
        )
    else:
        tri_accs = []

    tri_summary = []
    for (i, j, k), a in zip(triplets_1p2, tri_accs):
        # block pair is {j, j+1} which equals (min(j,k), max(j,k)); since k = j+1, min=j, max=k
        block_key = (j, k) if j < k else (k, j)
        a_anchor = float(solo_accs[i])
        a_block = float(pair_acc_map[block_key])
        tri_summary.append(
            {
                "anchor_band": int(i),
                "block_bands": [int(j), int(k)],
                "acc": float(a),
                "anchor_solo_acc": a_anchor,
                "block_pair_acc": a_block,
                "delta_over_anchor": float(a - a_anchor),
                "delta_over_block": float(a - a_block),
            }
        )

    tri_summary_path = os.path.join(
        args.results_dir, f"{args.prefix}_pair_1p2_summary.json"
    )
    with open(tri_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "triplets": tri_summary,
                "config": common_config,
            },
            f,
            indent=4,
        )
    print(
        f"[run_band_pair_probe] 1+2 triplet summary saved: {tri_summary_path}",
        flush=True,
    )

    _print_summary(
        solo_accs, cum_accs, prefix_delta,
        order_by_solo_desc, order_by_prefix_delta_desc, spearman_rho,
    )


def _print_summary(
    solo_accs: Sequence[float],
    cum_accs: Sequence[float],
    prefix_delta: Sequence[float],
    order_by_solo_desc: Sequence[int],
    order_by_prefix_delta_desc: Sequence[int],
    spearman_rho: float,
) -> None:
    print("\n==== Summary ====", flush=True)
    print(
        f"solo_acc        : {[round(float(a), 4) for a in solo_accs]}",
        flush=True,
    )
    print(
        f"cumulative_acc  : {[round(float(a), 4) for a in cum_accs]}",
        flush=True,
    )
    print(
        f"prefix_delta    : {[round(float(a), 4) for a in prefix_delta]}",
        flush=True,
    )
    top_solo = order_by_solo_desc[: min(10, len(order_by_solo_desc))]
    top_delta = order_by_prefix_delta_desc[: min(10, len(order_by_prefix_delta_desc))]
    print(
        f"top-10 by solo          : "
        f"{[f'B{i}={float(solo_accs[i]):.3f}' for i in top_solo]}",
        flush=True,
    )
    print(
        f"top-10 by prefix_delta  : "
        f"{[f'B{i}={float(prefix_delta[i]):.3f}' for i in top_delta]}",
        flush=True,
    )
    rho_str = "nan" if math.isnan(spearman_rho) else f"{spearman_rho:.3f}"
    print(f"Spearman(solo, prefix_delta): {rho_str}", flush=True)


if __name__ == "__main__":
    main()
