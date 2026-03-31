import numpy as np
import pandas as pd

# --- Constants ---
EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"

def calculate_embedding_statistics():
    """Calculates and prints the mean and std of embeddings for each genre."""
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    with np.load(EMBEDDINGS_PATH) as data:
        embeddings = data['embeddings']
        genres = data['genres']

    # Flatten the embeddings for statistical analysis
    # Shape (7997, 768, 2, 32) -> (7997, 49152)
    embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)

    df = pd.DataFrame({
        'genre': genres,
    })

    print("\n--- Embedding Statistics per Genre ---")
    for genre_name in np.unique(genres):
        genre_mask = (df['genre'] == genre_name)
        genre_embeddings = embeddings_flat[genre_mask]
        
        mean_val = np.mean(genre_embeddings)
        std_val = np.std(genre_embeddings)
        
        print(f"Genre: {genre_name}")
        print(f"  - Mean: {mean_val:.4f}")
        print(f"  - Std Dev: {std_val:.4f}")
        print("-" * 20)

if __name__ == "__main__":
    calculate_embedding_statistics()
