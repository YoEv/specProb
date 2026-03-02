import numpy as np

def load_data(path):
    with np.load(path) as data:
        X, y_str = data['embeddings'], data['genres']
    X = X.reshape(X.shape[0], -1, X.shape[-1])
    return X, y_str
