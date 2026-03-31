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
# Embeddings for the D_asap_100 classical composer task (Task 4).
EMBEDDINGS_ASAP_FILE = os.path.join(OUTPUT_DIR, "clap_embeddings_asap_t32.npz")
# Recommended path for future multi-layer embeddings (see extract_embeddings_multi_layer).
EMBEDDINGS_MULTI_FILE = os.path.join(OUTPUT_DIR, "clap_embeddings_t64_multilayer.npz")
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
            # Embedding layer: last_hidden_state = final (top) layer of CLAP's audio encoder (HTSAT).
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


def extract_embeddings_asap_composer(audio_dir, model, processor, device, sr=SAMPLE_RATE):
    """
    Extract CLAP embeddings for the D_asap_100 classical composer dataset.

    Assumptions:
        - audio_dir points to a directory containing 15s WAV files from ASAP,
          named like "Bach_007_...wav", "Beethoven_003_...wav", etc.
        - We only use 10 seconds from each file (centre crop) as input to CLAP.
        - CLAP's default 10s processing yields last_hidden_state of shape
          (1, 768, 2, 32). We do NOT add any extra 10s segment; this dataset
          does not use 10s+10s concat.

    Returns
    -------
    embeddings : np.ndarray, shape (B, 768, 2, 32)
    composers  : list[str], length B
    file_paths : list[str], length B
    """
    import glob

    model.to(device)
    model.eval()

    pattern = os.path.join(audio_dir, "*.wav")
    wav_paths = sorted(glob.glob(pattern))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {audio_dir}")

    embeddings_list = []
    composers = []
    file_paths = []

    print(f"[extract_asap_composer] Found {len(wav_paths)} WAV files in {audio_dir}")

    for audio_path in tqdm(wav_paths, desc="Extracting ASAP composer embeddings"):
        try:
            waveform, file_sr = librosa.load(audio_path, sr=sr, mono=True)
            # Centre crop to 10 seconds if longer than 10s, otherwise pad.
            target_len = sr * 10
            if len(waveform) > target_len:
                start = (len(waveform) - target_len) // 2
                waveform = waveform[start : start + target_len]
            elif len(waveform) < target_len:
                pad = target_len - len(waveform)
                waveform = np.pad(waveform, (0, pad), mode="constant")

            inputs = processor(
                text=None,
                audio=waveform,
                return_tensors="pt",
                sampling_rate=sr,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                features = model.get_audio_features(**inputs).last_hidden_state.cpu().numpy()

            # features is expected to be (1, 768, 2, 32) for a single 10s clip.
            embeddings_list.append(features)

            # Composer name is the prefix before the first underscore, e.g. "Bach".
            basename = os.path.basename(audio_path)
            composer = basename.split("_", 1)[0]
            composers.append(composer)
            file_paths.append(audio_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("No embeddings were extracted for ASAP composer dataset.")

    embeddings = np.vstack(embeddings_list)  # (B, 768, 2, 32)
    return embeddings, composers, file_paths


def extract_embeddings_multi_layer(metadata_df, model, processor, device, layer_indices=None):
    """
    Extract CLAP embeddings for multiple encoder layers per audio file.

    This is a multi-layer generalisation of `extract_embeddings` used for Task 1
    (plan 1.2: “不同层 / 不同模型的 embedding” 对比）。

    Shape conventions:
        raw hidden_states per segment: (T, D=768)
        two segments are concatenated in time → length 64.
        We stack selected layers, so final embeddings have shape:

            (B, L, 768, 2, 64)

        where B is number of tracks, L is number of selected layers.

    Args
    ----
    metadata_df : DataFrame with at least ['track_id', 'genre', 'audio_path'].
    model, processor, device : same as `extract_embeddings`.
    layer_indices : list[int] or None
        Indices into `hidden_states` to use. If None, defaults to
        [0, -1] (first and last layer). You can change this list
        to e.g. [0, 6, 11] once you know CLAP's depth.

    Returns
    -------
    embeddings : np.ndarray, shape (B, L, 768, 2, 64)
    layers : np.ndarray, shape (L,)
        Resolved layer indices actually used.
    genres_list : list[str]
    track_ids_list : list[int]
    """
    model.to(device)
    model.eval()

    embeddings_list = []
    genres_list = []
    track_ids_list = []
    resolved_layers = None

    print(f"Starting multi-layer feature extraction for {len(metadata_df)} tracks...")
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        audio_path = row["audio_path"]
        try:
            full_waveform, sr = librosa.load(
                audio_path, sr=SAMPLE_RATE, mono=True, duration=30
            )

            segment1 = full_waveform[: SAMPLE_RATE * 10]
            segment2 = full_waveform[SAMPLE_RATE * 10 : SAMPLE_RATE * 20]

            if len(segment2) < SAMPLE_RATE * 10:
                segment2 = np.pad(
                    segment2,
                    (0, SAMPLE_RATE * 10 - len(segment2)),
                    "constant",
                )

            def _encode_segments(wav):
                inputs = processor(
                    text=None,
                    audio=wav,
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.get_audio_features(
                        **inputs, output_hidden_states=True
                    )
                # outputs.hidden_states: tuple (num_layers+1, 1, T, D)
                hidden_states = outputs.hidden_states
                # squeeze batch dim, return list of (T, D)
                hs = [h.squeeze(0).cpu().numpy() for h in hidden_states]
                return hs

            hs1 = _encode_segments(segment1)
            hs2 = _encode_segments(segment2)

            num_layers_total = len(hs1)
            if layer_indices is None:
                # default: first and last encoder layer
                candidate = [0, num_layers_total - 1]
            else:
                candidate = list(layer_indices)
            # resolve negative indices, clip to valid range
            resolved_layers = [
                (li if li >= 0 else num_layers_total + li) for li in candidate
            ]
            resolved_layers = sorted(set(resolved_layers))

            sample_layers = []
            for li in resolved_layers:
                feat1 = hs1[li]  # (T1, D)
                feat2 = hs2[li]  # (T2, D)
                # we expect T1 == T2 == 32 for CLAP; enforce via trunc/pad if needed
                T_target = 32
                def _fix_length(x):
                    if x.shape[0] > T_target:
                        return x[:T_target]
                    if x.shape[0] < T_target:
                        pad = np.zeros((T_target - x.shape[0], x.shape[1]), dtype=x.dtype)
                        return np.concatenate([x, pad], axis=0)
                    return x
                feat1 = _fix_length(feat1)
                feat2 = _fix_length(feat2)
                # transpose to (D, T) then stack along segment axis
                seg1 = feat1.T  # (D, T)
                seg2 = feat2.T  # (D, T)
                # final per-layer shape: (D, 2, T_total=64)
                concatenated = np.stack([seg1, seg2], axis=1)  # (D, 2, 32)
                # concatenate along time inside the last dim
                concatenated = np.concatenate(
                    [concatenated[:, 0, :], concatenated[:, 1, :]], axis=-1
                )  # (D, 64)
                concatenated = concatenated.reshape(concatenated.shape[0], 2, 32)
                sample_layers.append(concatenated)

            # (L, 768, 2, 32)
            sample_layers = np.stack(sample_layers, axis=0)
            embeddings_list.append(sample_layers)
            genres_list.append(row["genre"])
            track_ids_list.append(row["track_id"])

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    embeddings = np.stack(embeddings_list, axis=0)  # (B, L, 768, 2, 32)
    layers_arr = np.array(resolved_layers if resolved_layers is not None else [])
    print(f"Multi-layer embeddings shape: {embeddings.shape}, layers: {layers_arr}")

    return embeddings, layers_arr, genres_list, track_ids_list

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
