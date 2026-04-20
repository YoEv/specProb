"""
Compute spectral probing profiles using PaSST embeddings NPZ files.

This script is a light-weight reimplementation of the CLAP-based spectral probing,
but operating directly on PaSST timestamp embeddings saved as:

    - embeddings: np.ndarray, shape (B, F_passt, T)
    - labels:     np.ndarray[str], e.g. 'genres' or 'composers'

It will:
    1. Load the NPZ.
    2. Run rFFT along the time axis.
    3. For each class (genre / composer) with enough samples, train a one-vs-rest
       logistic regression probe on flattened spectral features.
    4. Derive the learned spectral profile (weight per frequency) and save:
         - a JSON summary with spectral profiles and accuracies;
         - a PNG plot visualising mean magnitude and learned weight curves.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Ensure project root is importable when script is launched outside repo root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


MIN_SAMPLES_PER_CLASS = 10
RESULTS_DIR_DEFAULT = "results/passt_spectral"


def _load_passt_npz(path: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise KeyError(f"NPZ at {path} must contain 'embeddings' array.")
    if label_key not in data.files:
        raise KeyError(
            f"NPZ at {path} does not contain label key '{label_key}'. "
            f"Available keys: {list(data.files)}"
        )
    X = data["embeddings"]  # (B, F, T)
    y = data[label_key].astype(str)
    if X.ndim != 3:
        raise ValueError(f"Expected embeddings of shape (B, F, T), got {X.shape}")
    return X, y


def _load_passt_npz_many(paths: List[str], label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple shard NPZ files and concatenate along batch dimension.
    If T differs across shards, right-pad to the global max T.
    """
    if not paths:
        raise ValueError("No NPZ paths provided.")

    chunks_x: List[np.ndarray] = []
    chunks_y: List[np.ndarray] = []
    max_t = 0
    feat_dim = None

    for p in paths:
        X_i, y_i = _load_passt_npz(p, label_key=label_key)
        if feat_dim is None:
            feat_dim = int(X_i.shape[1])
        elif int(X_i.shape[1]) != feat_dim:
            raise ValueError(
                f"Feature dim mismatch across NPZ files: expected F={feat_dim}, got {X_i.shape[1]} at {p}"
            )
        max_t = max(max_t, int(X_i.shape[2]))
        chunks_x.append(X_i)
        chunks_y.append(y_i)

    padded_x: List[np.ndarray] = []
    for X_i in chunks_x:
        if int(X_i.shape[2]) == max_t:
            padded_x.append(X_i)
            continue
        pad = np.zeros((X_i.shape[0], X_i.shape[1], max_t), dtype=X_i.dtype)
        pad[:, :, : X_i.shape[2]] = X_i
        padded_x.append(pad)

    X = np.concatenate(padded_x, axis=0)
    y = np.concatenate(chunks_y, axis=0).astype(str)
    return X, y


def _safe_suffix(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")


def _balanced_binary_class_weight(n_pos: int, n_neg: int) -> dict[int, float]:
    """
    Compute sklearn-style 'balanced' weights for binary labels {0, 1}.

    Formula matches compute_class_weight('balanced'):
        w_c = n_samples / (n_classes * n_c)
    """
    n_pos = int(max(n_pos, 1))
    n_neg = int(max(n_neg, 1))
    n_total = n_pos + n_neg
    return {
        0: float(n_total / (2.0 * n_neg)),
        1: float(n_total / (2.0 * n_pos)),
    }


def _compute_and_plot_profiles(
    X: np.ndarray,
    y: np.ndarray,
    label_name: str,
    out_dir: str,
    prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    B, F, T = X.shape
    # rFFT along time axis
    coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)
    if coeffs.shape[2] != (T // 2 + 1):
        # Not fatal, but warn in case conventions change.
        print(
            f"[run_passt_spectral] Warning: coeffs.shape[2]={coeffs.shape[2]} "
            f"!= T//2+1={T//2+1}"
        )
    n_coeffs = coeffs.shape[2]
    n_features = F

    classes = sorted(set(y.tolist()))
    counts = {c: int((y == c).sum()) for c in classes}
    print(f"[run_passt_spectral] Found {len(classes)} {label_name}: {counts}")

    X_flat = coeffs.reshape(B, -1)

    mean_magnitude = {}
    learned_profiles = {}
    probe_accuracies = {}

    for cls in classes:
        mask = y == cls
        n_cls = int(mask.sum())
        if n_cls < MIN_SAMPLES_PER_CLASS:
            print(
                f"[run_passt_spectral] Skipping {label_name} '{cls}': "
                f"only {n_cls} samples (< {MIN_SAMPLES_PER_CLASS})."
            )
            continue

        print(
            f"[run_passt_spectral] Processing {label_name}='{cls}' "
            f"(n={n_cls})..."
        )

        # Mean magnitude over samples of this class and feature dims.
        mean_mag = coeffs[mask].mean(axis=(0, 1))  # (n_coeffs,)
        mean_magnitude[cls] = mean_mag.tolist()

        # One-vs-rest binary labels.
        y_binary = (y == cls).astype(int)
        if np.sum(y_binary) < MIN_SAMPLES_PER_CLASS or len(np.unique(y_binary)) < 2:
            print(
                f"[run_passt_spectral] Skipping probing for {cls}: "
                f"insufficient positives or only one class."
            )
            continue

        acc, final_model, _ = run_probe(X_flat, y_binary)
        probe_accuracies[cls] = float(acc)

        prof = learned_weight_profile(
            final_model, n_coeffs=n_coeffs, n_features=n_features
        )
        learned_profiles[cls] = prof.tolist()

        # Plot per-class spectral profile (mean magnitude + learned weight).
        x = np.arange(n_coeffs)
        fig, (ax_mag, ax_w) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1]}
        )

        ax_mag.plot(x, mean_mag, label=f"{cls} (n={n_cls})")
        ax_mag.set_xlabel("FFT coefficient index")
        ax_mag.set_ylabel("Mean magnitude")
        ax_mag.legend()
        ax_mag.set_title(f"Mean FFT magnitude — {cls}")

        ax_w.plot(x, prof, label=f"{cls} (acc={acc:.3f})")
        ax_w.set_xlabel("FFT coefficient index")
        ax_w.set_ylabel("Normalized weight")
        ax_w.set_ylim(0.0, 1.0)
        ax_w.legend()
        ax_w.set_title(f"Spectral profile (learned weight) — {cls}")

        fig.suptitle(
            f"PaSST spectral probing — {label_name}='{cls}' "
            f"(F={F}, T={T}, n_coeffs={n_coeffs})",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        cls_suffix = _safe_suffix(cls)
        png_path = os.path.join(
            out_dir, f"{prefix}_spectral_profile_{label_name}_{cls_suffix}.png"
        )
        plt.savefig(png_path)
        plt.close(fig)
        print(f"[run_passt_spectral] Saved plot to {png_path}")

    summary = {
        "shape": {"B": int(B), "F": int(F), "T": int(T), "n_coeffs": int(n_coeffs)},
        "label_name": label_name,
        "class_counts": counts,
        "probe_accuracies": probe_accuracies,
        "mean_magnitude_profiles": mean_magnitude,
        "learned_weight_profiles": learned_profiles,
    }
    json_path = os.path.join(out_dir, f"{prefix}_spectral_summary_{label_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_passt_spectral] Saved JSON summary to {json_path}")


def _streaming_spectral_profiles(
    npz_paths: List[str],
    label_key: str,
    out_dir: str,
    prefix: str,
    random_state: int = 42,
    test_fraction: float = 0.2,
    export_probe_metrics: bool = False,
    band_width: int = 4,
) -> None:
    """
    Memory-friendly variant for many large shard files.
    Trains one-vs-rest probes with incremental SGD and incremental scaling.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    # Pass 1: class counts + shape info
    class_counts = {}
    n_features = None
    n_coeffs = None
    max_t = 0
    total_b = 0
    for p in npz_paths:
        X_i, y_i = _load_passt_npz(p, label_key=label_key)
        total_b += int(X_i.shape[0])
        max_t = max(max_t, int(X_i.shape[2]))
        for c in y_i.tolist():
            class_counts[c] = class_counts.get(c, 0) + 1
        coeffs_i = apply_transform(X_i[:1], transform_type="fft", axis=2, window_type=None)
        if n_features is None:
            n_features = int(coeffs_i.shape[1])
            n_coeffs = int(coeffs_i.shape[2])
        elif int(coeffs_i.shape[1]) != n_features:
            raise ValueError(f"Feature mismatch in {p}")
        del X_i, y_i, coeffs_i

    assert n_features is not None and n_coeffs is not None
    print(f"[run_passt_spectral] Streaming mode: B~{total_b}, F={n_features}, n_coeffs={n_coeffs}")
    print(f"[run_passt_spectral] Found {len(class_counts)} labels: {class_counts}")

    eligible_classes = sorted([c for c, n in class_counts.items() if n >= MIN_SAMPLES_PER_CLASS])
    mean_magnitude = {}
    learned_profiles = {}
    probe_accuracies = {}
    streaming_probe_metrics = {}
    if export_probe_metrics:
        num_bands = n_coeffs // int(band_width)
        if num_bands <= 0:
            raise ValueError(
                f"band_width={band_width} too large for n_coeffs={n_coeffs} in streaming mode."
            )
        band_indices = {
            b: list(range(b * band_width, (b + 1) * band_width))
            for b in range(num_bands)
        }

    for cls in eligible_classes:
        print(f"[run_passt_spectral] Streaming class '{cls}' (n={class_counts[cls]})...")
        n_pos = int(class_counts[cls])
        n_neg = int(total_b - n_pos)
        class_weight = _balanced_binary_class_weight(n_pos=n_pos, n_neg=n_neg)
        scaler = StandardScaler()
        clf = SGDClassifier(
            loss="log_loss",
            random_state=random_state,
            class_weight=class_weight,
            max_iter=1,
            learning_rate="optimal",
            tol=None,
        )
        if export_probe_metrics:
            band_models = [
                {
                    "scaler": StandardScaler(),
                    "clf": SGDClassifier(
                        loss="log_loss",
                        random_state=random_state,
                        class_weight=class_weight,
                        max_iter=1,
                        learning_rate="optimal",
                        tol=None,
                    ),
                    "initialized": False,
                    "correct": 0,
                    "total": 0,
                }
                for _ in range(num_bands)
            ]
            cum_models = [
                {
                    "scaler": StandardScaler(),
                    "clf": SGDClassifier(
                        loss="log_loss",
                        random_state=random_state,
                        class_weight=class_weight,
                        max_iter=1,
                        learning_rate="optimal",
                        tol=None,
                    ),
                    "initialized": False,
                    "correct": 0,
                    "total": 0,
                }
                for _ in range(num_bands)
            ]

        mag_sum = np.zeros((n_coeffs,), dtype=np.float64)
        mag_count = 0
        correct = 0
        total = 0
        initialized = False

        for p in npz_paths:
            X_i, y_i = _load_passt_npz(p, label_key=label_key)
            coeffs_i = apply_transform(X_i, transform_type="fft", axis=2, window_type=None)
            X_flat_i = coeffs_i.reshape(coeffs_i.shape[0], -1).astype(np.float32, copy=False)
            y_bin_i = (y_i == cls).astype(int)

            mask_cls = y_bin_i == 1
            if np.any(mask_cls):
                mag_sum += coeffs_i[mask_cls].mean(axis=1).sum(axis=0)
                mag_count += int(mask_cls.sum())

            if len(np.unique(y_bin_i)) < 2:
                del X_i, y_i, coeffs_i, X_flat_i, y_bin_i
                continue

            test_mask = rng.random(X_flat_i.shape[0]) < test_fraction
            train_mask = ~test_mask
            if not np.any(train_mask):
                del X_i, y_i, coeffs_i, X_flat_i, y_bin_i, test_mask, train_mask
                continue

            X_train = X_flat_i[train_mask]
            y_train = y_bin_i[train_mask]
            scaler.partial_fit(X_train)
            X_train_s = scaler.transform(X_train)

            if not initialized:
                clf.partial_fit(X_train_s, y_train, classes=np.array([0, 1], dtype=np.int64))
                initialized = True
            else:
                clf.partial_fit(X_train_s, y_train)

            if np.any(test_mask):
                X_test = scaler.transform(X_flat_i[test_mask])
                y_test = y_bin_i[test_mask]
                y_pred = clf.predict(X_test)
                correct += int((y_pred == y_test).sum())
                total += int(y_test.size)

            if export_probe_metrics:
                coeffs_train = coeffs_i[train_mask]
                coeffs_test = coeffs_i[test_mask] if np.any(test_mask) else None
                y_test = y_bin_i[test_mask] if np.any(test_mask) else None

                # Single-band streaming probes.
                for b in range(num_bands):
                    bm = band_models[b]
                    Xb_train = coeffs_train[:, :, band_indices[b]].reshape(coeffs_train.shape[0], -1)
                    bm["scaler"].partial_fit(Xb_train)
                    Xb_train_s = bm["scaler"].transform(Xb_train)
                    if not bm["initialized"]:
                        bm["clf"].partial_fit(
                            Xb_train_s,
                            y_train,
                            classes=np.array([0, 1], dtype=np.int64),
                        )
                        bm["initialized"] = True
                    else:
                        bm["clf"].partial_fit(Xb_train_s, y_train)

                    if coeffs_test is not None and y_test is not None and y_test.size > 0:
                        Xb_test = coeffs_test[:, :, band_indices[b]].reshape(coeffs_test.shape[0], -1)
                        Xb_test_s = bm["scaler"].transform(Xb_test)
                        yb_pred = bm["clf"].predict(Xb_test_s)
                        bm["correct"] += int((yb_pred == y_test).sum())
                        bm["total"] += int(y_test.size)

                # Cumulative-prefix streaming probes.
                for b in range(num_bands):
                    cm = cum_models[b]
                    end_coeff = int((b + 1) * band_width)
                    Xc_train = coeffs_train[:, :, :end_coeff].reshape(coeffs_train.shape[0], -1)
                    cm["scaler"].partial_fit(Xc_train)
                    Xc_train_s = cm["scaler"].transform(Xc_train)
                    if not cm["initialized"]:
                        cm["clf"].partial_fit(
                            Xc_train_s,
                            y_train,
                            classes=np.array([0, 1], dtype=np.int64),
                        )
                        cm["initialized"] = True
                    else:
                        cm["clf"].partial_fit(Xc_train_s, y_train)

                    if coeffs_test is not None and y_test is not None and y_test.size > 0:
                        Xc_test = coeffs_test[:, :, :end_coeff].reshape(coeffs_test.shape[0], -1)
                        Xc_test_s = cm["scaler"].transform(Xc_test)
                        yc_pred = cm["clf"].predict(Xc_test_s)
                        cm["correct"] += int((yc_pred == y_test).sum())
                        cm["total"] += int(y_test.size)

            del X_i, y_i, coeffs_i, X_flat_i, y_bin_i, test_mask, train_mask, X_train, y_train, X_train_s

        if not initialized:
            print(f"[run_passt_spectral] Skipping {cls}: insufficient trainable chunks.")
            continue

        acc = float(correct / total) if total > 0 else 0.0
        probe_accuracies[cls] = acc

        if mag_count > 0:
            mean_mag = (mag_sum / float(mag_count)).astype(np.float64)
        else:
            mean_mag = np.zeros((n_coeffs,), dtype=np.float64)
        mean_magnitude[cls] = mean_mag.tolist()

        prof = learned_weight_profile(clf, n_coeffs=n_coeffs, n_features=n_features)
        learned_profiles[cls] = prof.tolist()

        x = np.arange(n_coeffs)
        fig, (ax_mag, ax_w) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1]})
        ax_mag.plot(x, mean_mag, label=f"{cls} (n={class_counts[cls]})")
        ax_mag.set_xlabel("FFT coefficient index")
        ax_mag.set_ylabel("Mean magnitude")
        ax_mag.legend()
        ax_mag.set_title(f"Mean FFT magnitude — {cls}")

        ax_w.plot(x, prof, label=f"{cls} (acc={acc:.3f})")
        ax_w.set_xlabel("FFT coefficient index")
        ax_w.set_ylabel("Normalized weight")
        ax_w.set_ylim(0.0, 1.0)
        ax_w.legend()
        ax_w.set_title(f"Spectral profile (learned weight) — {cls}")

        fig.suptitle(
            f"PaSST spectral probing (stream) — {label_key.rstrip('s')}='{cls}' "
            f"(F={n_features}, n_coeffs={n_coeffs})",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        cls_suffix = _safe_suffix(cls)
        png_path = os.path.join(out_dir, f"{prefix}_spectral_profile_{label_key.rstrip('s')}_{cls_suffix}.png")
        plt.savefig(png_path)
        plt.close(fig)
        print(f"[run_passt_spectral] Saved plot to {png_path}")

        if export_probe_metrics:
            band_accuracies = [
                float(m["correct"] / m["total"]) if m["initialized"] and m["total"] > 0 else 0.0
                for m in band_models
            ]
            cumulative_accuracies = [
                float(m["correct"] / m["total"]) if m["initialized"] and m["total"] > 0 else 0.0
                for m in cum_models
            ]
            cumulative_auto = max(cumulative_accuracies) if cumulative_accuracies else 0.0

            stream_metrics = {
                "target_label": cls,
                "label_key": label_key,
                "transform_type": "fft",
                "mode": "streaming_incremental_sgd",
                "accuracies": {
                    "full_embedding": float(acc),
                    "band_specific": band_accuracies,
                    "cumulative_auto": float(cumulative_auto),
                    "cumulative_per_band": cumulative_accuracies,
                    "trained_auto": None,
                },
                "config": {
                    "n_coeffs": int(n_coeffs),
                    "n_features": int(n_features),
                    "band_width": int(band_width),
                    "npz_paths": npz_paths,
                    "results_dir": out_dir,
                    "prefix": prefix,
                    "streaming": True,
                    "test_fraction": float(test_fraction),
                    "random_state": int(random_state),
                },
            }
            streaming_probe_metrics[cls] = stream_metrics

            metrics_path = os.path.join(
                out_dir,
                f"{prefix}_spectral_summary_{cls_suffix}_fft_metrics.json",
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(stream_metrics, f, indent=2)

            # Spectral-summary style figure: bars + learned profile
            bar_labels = ["ORIG"] + [f"B{b}" for b in range(num_bands)] + ["AUTO_CMAX"]
            bar_vals = [float(acc)] + band_accuracies + [float(cumulative_auto)]
            colors = ["gray"] + ["lightskyblue"] * num_bands + ["mediumpurple"]
            x2 = np.arange(n_coeffs)
            fig2, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
            )
            fig2.suptitle(
                f"Streaming Spectral Probing ({prefix}) — {label_key.rstrip('s')}={cls}",
                fontsize=14,
            )
            ax1.bar(bar_labels, bar_vals, color=colors)
            ax1.set_ylim(0, 1)
            ax1.set_ylabel("Accuracy")
            chance = float(class_counts[cls] / total_b)
            ax1.axhline(y=chance, color="r", linestyle="--", label=f"Chance ({chance:.3f})")
            ax1.set_title("Probe Performance (bands)")
            ax1.tick_params(axis="x", rotation=45)
            ax1.legend()

            ax2.plot(x2, prof, color="black", linewidth=1.2)
            ax2.set_xlabel("Frequency Coefficient")
            ax2.set_ylabel("Normalized weight")
            ax2.set_ylim(0, 1)
            ax2.set_title("Spectral Profile")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            summary_png = os.path.join(
                out_dir,
                f"{prefix}_spectral_summary_{cls_suffix}_fft.png",
            )
            plt.savefig(summary_png)
            plt.close(fig2)
            print(f"[run_passt_spectral] Saved streaming summary plot to {summary_png}")

    summary = {
        "shape": {"B": int(total_b), "F": int(n_features), "T_max": int(max_t), "n_coeffs": int(n_coeffs)},
        "label_name": label_key.rstrip("s"),
        "class_counts": class_counts,
        "probe_accuracies": probe_accuracies,
        "mean_magnitude_profiles": mean_magnitude,
        "learned_weight_profiles": learned_profiles,
        "mode": "streaming_incremental_sgd",
        "streaming_probe_metrics": streaming_probe_metrics,
    }
    json_path = os.path.join(out_dir, f"{prefix}_spectral_summary_{label_key.rstrip('s')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_passt_spectral] Saved JSON summary to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run spectral probing and plot spectral profiles from a PaSST "
            "embeddings NPZ."
        )
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default=None,
        help=(
            "Path to PaSST embeddings NPZ "
            "(e.g. data_artifacts/passt_embeddings_t64.npz or "
            "data_artifacts/passt_embeddings_asap_t32.npz)."
        ),
    )
    parser.add_argument(
        "--npz_paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional multiple NPZ inputs (e.g. shard files). "
            "When set, these are loaded and concatenated along batch axis."
        ),
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="genres",
        help="Name of the label array inside the NPZ (e.g. 'genres' or 'composers').",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR_DEFAULT,
        help="Directory to store JSON summaries and plots.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="passt",
        help="Filename prefix for outputs (useful when running multiple configs).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Use shard-streaming mode (incremental training) for large multi-NPZ inputs "
            "to avoid OOM. Requires --npz_paths."
        ),
    )
    parser.add_argument(
        "--streaming_export_probe_metrics",
        action="store_true",
        help=(
            "In streaming mode, additionally train streaming band/cumulative probes and "
            "export spectral_summary-style metrics/plots per class."
        ),
    )
    parser.add_argument(
        "--streaming_band_width",
        type=int,
        default=4,
        help="Band width for streaming probe metrics export (default: 4).",
    )
    args = parser.parse_args()
    if not args.npz_path and not args.npz_paths:
        parser.error("Provide --npz_path (single file) or --npz_paths (multiple files).")
    if args.npz_path and args.npz_paths:
        parser.error("Use either --npz_path or --npz_paths, not both.")
    if args.streaming and not args.npz_paths:
        parser.error("--streaming requires --npz_paths.")

    os.makedirs(args.results_dir, exist_ok=True)

    if args.npz_paths and args.streaming:
        print(
            f"[run_passt_spectral] Streaming embeddings from {len(args.npz_paths)} NPZ files "
            f"(label_key='{args.label_key}')"
        )
        for p in args.npz_paths:
            print(f"  - {p}")
        _streaming_spectral_profiles(
            npz_paths=args.npz_paths,
            label_key=args.label_key,
            out_dir=args.results_dir,
            prefix=args.prefix,
            export_probe_metrics=args.streaming_export_probe_metrics,
            band_width=args.streaming_band_width,
        )
        return
    elif args.npz_paths:
        print(
            f"[run_passt_spectral] Loading embeddings from {len(args.npz_paths)} NPZ files "
            f"(label_key='{args.label_key}')"
        )
        for p in args.npz_paths:
            print(f"  - {p}")
        X, y = _load_passt_npz_many(args.npz_paths, label_key=args.label_key)
    else:
        print(
            f"[run_passt_spectral] Loading embeddings from {args.npz_path} "
            f"(label_key='{args.label_key}')"
        )
        X, y = _load_passt_npz(args.npz_path, label_key=args.label_key)
    print(f"[run_passt_spectral] X.shape={X.shape}, n_labels={len(y)}")

    _compute_and_plot_profiles(
        X=X,
        y=y,
        label_name=args.label_key.rstrip("s"),
        out_dir=args.results_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

