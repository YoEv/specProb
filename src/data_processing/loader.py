import numpy as np


def load_data(path):
    """
    Load embeddings and genre labels. Reshape is fixed project-wide.

    NPZ keys: 'embeddings', 'genres', ('track_ids').
    Embeddings raw shape: (B, 768, 2, 64) — two segments concatenated in time (2*32 → 64).
    Output X shape: (B, 1536, 64) = (B, 768*2, 64). FFT on axis=2 (length 64) yields n_coeffs=33.
    """
    with np.load(path) as data:
        X, y_str = data['embeddings'], data['genres']
    # (B, 768, 2, 64) → (B, 1536, 64); all downstream code assumes this.
    X = X.reshape(X.shape[0], -1, X.shape[-1])
    return X, y_str
