"""
Learned weight profile (spectral profile) from a trained linear probe.
Used for plotting "weight per frequency" in spectral probing figures.
"""
import numpy as np


def _resolve_weight_matrix(model, n_coeffs: int, n_features: int | None = None) -> np.ndarray:
    """Return absolute probe weights shaped as (n_features, n_coeffs)."""
    raw = model.coef_.flatten()
    w = np.abs(raw)
    if n_features is None:
        n_features = w.size // n_coeffs
    if n_features * n_coeffs != w.size:
        raise ValueError(
            f"Cannot reshape weights: got {w.size} values, "
            f"n_features={n_features}, n_coeffs={n_coeffs}."
        )
    return w.reshape(n_features, n_coeffs)


def learned_weight_profile(model, n_coeffs: int, n_features: int = None) -> np.ndarray:
    """
    From a fitted logistic regression probe, compute the per-frequency learned weight
    (average |coef_| over feature dimension), normalized to [0, 1].

    This is the "Spectral Profile" / "weight per frequency" in the paper: which
    frequency coefficients the probe relies on most.

    Parameters
    ----------
    model : sklearn LogisticRegression (or any with .coef_ of shape (1, n_features * n_coeffs))
        Fitted probe trained on flattened spectral features (n_samples, n_features * n_coeffs).
    n_coeffs : int
        Number of frequency coefficients (e.g. 33 for rfft(64)).
    n_features : int, optional
        Number of feature dimensions (e.g. 1536). If None, inferred as coef_.size // n_coeffs.

    Returns
    -------
    np.ndarray of shape (n_coeffs,)
        Normalized weight in [0, 1] per frequency coefficient.
    """
    w_per_freq = np.mean(_resolve_weight_matrix(model, n_coeffs, n_features), axis=0)
    if w_per_freq.max() == w_per_freq.min():
        return np.zeros_like(w_per_freq)
    return (w_per_freq - w_per_freq.min()) / (w_per_freq.max() - w_per_freq.min())


def learned_weight_profile_stats(
    model, n_coeffs: int, n_features: int = None
) -> dict[str, np.ndarray]:
    """
    Compute per-frequency summary stats across feature dimensions.

    Returns both raw and normalized values. Normalization uses the min-max range
    of the mean curve to keep all statistics in a comparable 0-1 scale.
    """
    w2d = _resolve_weight_matrix(model, n_coeffs, n_features)  # (F, K)
    mean_raw = np.mean(w2d, axis=0)
    std_raw = np.std(w2d, axis=0)
    median_raw = np.median(w2d, axis=0)
    q25_raw = np.percentile(w2d, 25, axis=0)
    q75_raw = np.percentile(w2d, 75, axis=0)

    spread = float(mean_raw.max() - mean_raw.min())
    if spread <= 0:
        mean_norm = np.zeros_like(mean_raw)
        std_norm = np.zeros_like(std_raw)
        median_norm = np.zeros_like(median_raw)
        q25_norm = np.zeros_like(q25_raw)
        q75_norm = np.zeros_like(q75_raw)
    else:
        offset = float(mean_raw.min())
        mean_norm = (mean_raw - offset) / spread
        std_norm = std_raw / spread
        median_norm = (median_raw - offset) / spread
        q25_norm = (q25_raw - offset) / spread
        q75_norm = (q75_raw - offset) / spread

    return {
        "mean_raw": mean_raw,
        "std_raw": std_raw,
        "median_raw": median_raw,
        "q25_raw": q25_raw,
        "q75_raw": q75_raw,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "median_norm": median_norm,
        "q25_norm": q25_norm,
        "q75_norm": q75_norm,
    }


def _resolve_weight_matrix_multiclass(
    model, n_coeffs: int, n_features: int, n_classes: int
) -> np.ndarray:
    """Return absolute multiclass probe weights shaped as (C, F, K).

    Expects `model.coef_` of shape (C, F * K) (sklearn multinomial logistic
    regression or an equivalent torch linear head wrapped to expose `.coef_`).
    """
    raw = np.asarray(model.coef_)
    if raw.ndim != 2:
        raise ValueError(
            f"Expected coef_ of shape (C, D), got shape {raw.shape}."
        )
    C, D = raw.shape
    if C != n_classes:
        raise ValueError(
            f"coef_ has {C} rows but n_classes={n_classes}."
        )
    if n_features * n_coeffs != D:
        raise ValueError(
            f"Cannot reshape weights: D={D}, n_features={n_features}, "
            f"n_coeffs={n_coeffs}."
        )
    return np.abs(raw).reshape(C, n_features, n_coeffs)


def _minmax_norm(arr: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Min-max normalize ``arr`` using the min-max range of ``ref`` (defaults to arr)."""
    r = arr if ref is None else ref
    lo = float(np.min(r))
    hi = float(np.max(r))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def learned_weight_profile_multiclass(
    model, n_coeffs: int, n_features: int, n_classes: int
) -> np.ndarray:
    """Overall multiclass spectral profile (length n_coeffs), min-max normalized.

    profile[k] = mean_c mean_f |W[c, f, k]|, then min-max normalized.
    """
    w3d = _resolve_weight_matrix_multiclass(model, n_coeffs, n_features, n_classes)
    overall_mean = w3d.mean(axis=(0, 1))  # (K,)
    return _minmax_norm(overall_mean)


def learned_weight_profile_stats_multiclass(
    model,
    n_coeffs: int,
    n_features: int,
    n_classes: int,
    class_names: list[str],
) -> dict:
    """Per-frequency summary stats for an overall curve AND each per-class curve.

    - Overall curve: mean/std/percentiles across (C, F) axes for each frequency k.
    - Per-class curve: for each class c, mean/std/percentiles across F axis only.

    All `*_norm` variants are min-max normalized against that curve's own mean-raw
    range (offset = min(mean_raw), spread = max(mean_raw) - min(mean_raw)), so that
    each curve sits on [0, 1] independently.

    Returns:
        {
            "overall": {mean_raw, std_raw, median_raw, q25_raw, q75_raw,
                        mean_norm, std_norm, median_norm, q25_norm, q75_norm},
            "per_class": {cls_name: {same keys as overall}},
        }
    """
    w3d = _resolve_weight_matrix_multiclass(model, n_coeffs, n_features, n_classes)
    C, F, K = w3d.shape
    if len(class_names) != C:
        raise ValueError(
            f"class_names has {len(class_names)} entries but C={C}."
        )

    def _stats_from_2d(w2d: np.ndarray) -> dict:
        mean_raw = np.mean(w2d, axis=0)
        std_raw = np.std(w2d, axis=0)
        median_raw = np.median(w2d, axis=0)
        q25_raw = np.percentile(w2d, 25, axis=0)
        q75_raw = np.percentile(w2d, 75, axis=0)

        spread = float(mean_raw.max() - mean_raw.min())
        if spread <= 0:
            zero = np.zeros_like(mean_raw)
            return {
                "mean_raw": mean_raw,
                "std_raw": std_raw,
                "median_raw": median_raw,
                "q25_raw": q25_raw,
                "q75_raw": q75_raw,
                "mean_norm": zero,
                "std_norm": np.zeros_like(std_raw),
                "median_norm": zero,
                "q25_norm": zero,
                "q75_norm": zero,
            }
        offset = float(mean_raw.min())
        return {
            "mean_raw": mean_raw,
            "std_raw": std_raw,
            "median_raw": median_raw,
            "q25_raw": q25_raw,
            "q75_raw": q75_raw,
            "mean_norm": (mean_raw - offset) / spread,
            "std_norm": std_raw / spread,
            "median_norm": (median_raw - offset) / spread,
            "q25_norm": (q25_raw - offset) / spread,
            "q75_norm": (q75_raw - offset) / spread,
        }

    overall = _stats_from_2d(w3d.reshape(C * F, K))
    per_class = {cls: _stats_from_2d(w3d[i]) for i, cls in enumerate(class_names)}
    return {"overall": overall, "per_class": per_class}
