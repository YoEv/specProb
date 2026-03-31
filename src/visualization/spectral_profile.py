"""
Learned weight profile (spectral profile) from a trained linear probe.
Used for plotting "weight per frequency" in spectral probing figures.
"""
import numpy as np


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
    raw = model.coef_.flatten()
    w = np.abs(raw)
    if n_features is None:
        n_features = w.size // n_coeffs
    # (n_features, n_coeffs) -> mean over features -> (n_coeffs,)
    w_per_freq = np.mean(w.reshape(n_features, n_coeffs), axis=0)
    if w_per_freq.max() == w_per_freq.min():
        return np.zeros_like(w_per_freq)
    return (w_per_freq - w_per_freq.min()) / (w_per_freq.max() - w_per_freq.min())
