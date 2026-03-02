import os
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

# --- Constants ---
METADATA_PATH = "data_artifacts/fma_metadata.csv"
OUTPUT_DIR = "data_artifacts"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "clap_embeddings_t64.npz")
SAMPLE_RATE = 48000

# --- Model Loading ---
def load_model():
    """Loads the CLAP model and processor from Hugging Face."""
    print("Loading CLAP model...")
    try:
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        print("CLAP model loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error loading CLAP model: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed.")
        return None, None

# --- Feature Extraction ---
def extract_embeddings(metadata_df, model, processor, device):
    """Extracts CLAP embeddings for each audio file in the metadata."""
    model.to(device)
    model.eval()
    
    embeddings_list = []
    genres_list = []
    track_ids_list = []
    
    print(f"Starting feature extraction for {len(metadata_df)} tracks...")
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        audio_path = row['audio_path']
        try:
            # Load and resample audio to full length
            full_waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=30)

            # Define two 10-second segments
            segment1 = full_waveform[:SAMPLE_RATE * 10]
            segment2 = full_waveform[SAMPLE_RATE * 10:SAMPLE_RATE * 20]

            # Ensure the second segment is 10 seconds long, pad if necessary
            if len(segment2) < SAMPLE_RATE * 10:
                segment2 = np.pad(segment2, (0, SAMPLE_RATE * 10 - len(segment2)), 'constant')

            # Process segment 1
            inputs1 = processor(text=None, audio=segment1, return_tensors="pt", sampling_rate=SAMPLE_RATE)
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            with torch.no_grad():
                features1 = model.get_audio_features(**inputs1).last_hidden_state.cpu().numpy()

            # Process segment 2
            inputs2 = processor(text=None, audio=segment2, return_tensors="pt", sampling_rate=SAMPLE_RATE)
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            with torch.no_grad():
                features2 = model.get_audio_features(**inputs2).last_hidden_state.cpu().numpy()

            # Concatenate along the time axis (the last dimension)
            concatenated_features = np.concatenate((features1, features2), axis=-1)
            
            embeddings_list.append(concatenated_features)
            genres_list.append(row['genre'])
            track_ids_list.append(row['track_id'])
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
            
    return np.vstack(embeddings_list), genres_list, track_ids_list

# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    model, processor = load_model()
    
    if model and processor:
        # --- Load Metadata ---
        print(f"Loading metadata from {METADATA_PATH}...")
        if not os.path.exists(METADATA_PATH):
            print(f"Error: Metadata file not found at {METADATA_PATH}")
            print("Please run the data preparation script first.")
        else:
            metadata = pd.read_csv(METADATA_PATH)
            
            # --- Extract Features ---
            embeddings, genres, track_ids = extract_embeddings(metadata, model, processor, device)
            
            # --- Save Results ---
            if len(embeddings) > 0:
                print(f"Saving embeddings to {EMBEDDINGS_FILE}...")
                np.savez_compressed(
                    EMBEDDINGS_FILE,
                    embeddings=embeddings,
                    genres=np.array(genres),
                    track_ids=np.array(track_ids)
                )
                print("Embeddings saved successfully.")
            else:
                print("No embeddings were extracted. Nothing to save.")
