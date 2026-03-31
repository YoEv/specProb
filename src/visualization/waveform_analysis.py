import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- Constants ---
METADATA_PATH = "data_artifacts/fma_metadata.csv"
OUTPUT_DIR = "results/waveform_analysis"
N_SAMPLES = 3  # Number of samples per genre
TARGET_GENRE = 'Pop'
COMPARE_GENRE = 'Rock'
SAMPLE_RATE = 48000

def plot_waveform_and_spectrogram(track_id, audio_path, genre, output_dir):
    """Loads audio, and plots its waveform and Mel spectrogram."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Genre: {genre} - Track ID: {track_id}", fontsize=16)

        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Waveform")
        ax1.set_xlabel(None)

        # Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title("Mel Spectrogram")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f"{genre}_{track_id}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot to {save_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        exit()

    metadata = pd.read_csv(METADATA_PATH)

    # Get samples for target genre
    target_samples = metadata[metadata['genre'] == TARGET_GENRE].sample(n=N_SAMPLES, random_state=42)
    # Get samples for comparison genre
    compare_samples = metadata[metadata['genre'] == COMPARE_GENRE].sample(n=N_SAMPLES, random_state=42)

    print(f"--- Plotting for {TARGET_GENRE} ---")
    for _, row in target_samples.iterrows():
        plot_waveform_and_spectrogram(row['track_id'], row['audio_path'], TARGET_GENRE, OUTPUT_DIR)

    print(f"\n--- Plotting for {COMPARE_GENRE} ---")
    for _, row in compare_samples.iterrows():
        plot_waveform_and_spectrogram(row['track_id'], row['audio_path'], COMPARE_GENRE, OUTPUT_DIR)
