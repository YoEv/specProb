import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"

def load_and_prepare_data(file_path=EMBEDDINGS_PATH):
    """
    Loads embeddings and reshapes to the (B, 1536, 32) format.
    Returns the time-domain data and labels.
    """
    with np.load(file_path) as data:
        raw_embeddings = data['embeddings']
        genres = data['genres']
    # Reshape [B, 768, 2, 32] -> [B, 1536, 32]
    X = raw_embeddings.transpose(0, 2, 1, 3).reshape(raw_embeddings.shape[0], -1, 32)
    y = np.array(genres)
    print(f"Data loaded. Reshaped X to: {X.shape}")
    return X, y

def run_probe(X_flat, y, use_scaler=True):
    """
    Runs a standard linear probe using Logistic Regression.
    
    Args:
        X_flat (np.array): Flattened feature data of shape (n_samples, n_features).
        y (np.array): Labels.
        use_scaler (bool): Whether to apply StandardScaler to the features.

    Returns:
        float: The classification accuracy.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy
