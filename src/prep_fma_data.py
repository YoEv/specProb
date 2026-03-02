import os
import pandas as pd

# --- Constants ---
FMA_METADATA_DIR = "data/fma_metadata/fma_metadata"
FMA_AUDIO_DIR = "data/fma_small"
OUTPUT_ARTIFACTS_DIR = "data_artifacts"

# --- Functions ---

def load_tracks(metadata_dir):
    """Loads the tracks.csv file and returns a DataFrame."""
    tracks_path = os.path.join(metadata_dir, "tracks.csv")
    if not os.path.exists(tracks_path):
        print(f"Error: Metadata file not found at {tracks_path}")
        print("Please download fma_metadata.zip, unzip it, and place its contents in the 'data/fma_metadata/' directory.")
        return None
    
    try:
        tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
        return tracks
    except Exception as e:
        print(f"Error loading {tracks_path}: {e}")
        return None

def get_audio_path(track_id, audio_dir):
    """Constructs the audio file path for a given track ID."""
    tid_str = f"{track_id:06d}"
    return os.path.join(audio_dir, tid_str[:3], f"{tid_str}.mp3")

def create_metadata_file(tracks_df, audio_dir, output_path):
    """
    Creates a project-specific metadata CSV file.
    
    This file will contain track_id, genre, and the path to the audio file
    for all tracks in the 'small' subset that have a corresponding audio file.
    """
    small_subset = tracks_df[tracks_df[('set', 'subset')] == 'small']
    
    metadata_list = []
    for track_id, row in small_subset.iterrows():
        audio_path = get_audio_path(track_id, audio_dir)
        if os.path.exists(audio_path):
            genre = row[('track', 'genre_top')]
            if pd.notna(genre):
                metadata_list.append({
                    'track_id': track_id,
                    'genre': genre,
                    'audio_path': audio_path
                })
            else:
                print(f"Warning: Genre is missing for track {track_id}, skipping.")
        else:
            # This can be noisy, so we can choose to disable it if needed.
            # print(f"Warning: Audio file not found for track {track_id}, skipping.")
            pass

    project_metadata = pd.DataFrame(metadata_list)
    project_metadata.to_csv(output_path, index=False)
    
    print(f"Successfully created metadata file at {output_path}")
    print(f"Total tracks in metadata: {len(project_metadata)}")

# --- Main Execution ---

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_ARTIFACTS_DIR, exist_ok=True)
    
    print("Loading FMA metadata...")
    raw_tracks_df = load_tracks(FMA_METADATA_DIR)
    
    if raw_tracks_df is not None:
        output_metadata_path = os.path.join(OUTPUT_ARTIFACTS_DIR, "fma_metadata.csv")
        print(f"Creating project metadata file at {output_metadata_path}...")
        create_metadata_file(raw_tracks_df, FMA_AUDIO_DIR, output_metadata_path)
