import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    # Load metadata
    # Check if aligned or original exists
    aligned_csv = os.path.join(data_dir, "segments_aligned.csv")
    original_csv = os.path.join(data_dir, "segments.csv")
    
    if os.path.exists(aligned_csv):
        df = pd.read_csv(aligned_csv)
    else:
        df = pd.read_csv(original_csv)
        
    # Load embeddings
    emb_path = os.path.join(data_dir, "embeddings.npy")
    X = np.load(emb_path)
    
    return df, X

def train_probe(data_dir):
    df, X = load_data(data_dir)
    print(f"Data loaded: X shape {X.shape}, df shape {df.shape}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['chord_label'])
    print(f"Classes: {le.classes_}")
    
    # Split by Track ID
    track_ids = df['track_id'].unique()
    train_tracks, test_tracks = train_test_split(track_ids, test_size=0.2, random_state=42)
    
    print(f"Train tracks: {len(train_tracks)}, Test tracks: {len(test_tracks)}")
    
    # Create masks
    train_mask = df['track_id'].isin(train_tracks)
    test_mask = df['track_id'].isin(test_tracks)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train Logistic Regression
    # Increase max_iter for convergence
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs')
    print("Training Logistic Regression...")
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print("-" * 30)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("-" * 30)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # Save results
    results = {
        'model': 'CLAP (Frozen)',
        'accuracy': acc,
        'macro_f1': macro_f1,
        'test_tracks': test_tracks.tolist()
    }
    
    return results

if __name__ == "__main__":
    DATA_DIR = "/home/evev/specProb/data_artifacts"
    train_probe(DATA_DIR)
