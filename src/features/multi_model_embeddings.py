from __future__ import annotations

from typing import Literal, Sequence, Tuple

import numpy as np

# NOTE: We deliberately avoid importing heavy model libraries here (hear21passt,
# speechbrain, etc.) in the skeleton. Those should be imported lazily inside
# the concrete get_*_embeddings implementations once you wire them up.


ModelName = Literal["clap", "passt", "pann_s", "beats"]


def get_clap_embeddings(
    file_list: Sequence[str],
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for CLAP embedding extraction.

    In the current project, CLAP embeddings for FMA/ASAP are already computed
    and stored as NPZ; for sanity-check pipelines we typically do NOT want to
    recompute them here. This function is provided mainly for symmetry with
    other models and for future use if we add new datasets.

    Returns:
        embeddings: np.ndarray of shape (B, F, T)
        labels:     np.ndarray of shape (B,)
    """
    raise NotImplementedError(
        "get_clap_embeddings is a placeholder. For existing experiments, "
        "load CLAP NPZ files via src.config.spectral_experiments instead."
    )


def get_passt_embeddings(
    file_list: Sequence[str],
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for PaSST embedding extraction using the `hear21passt` package.

    Recommended pattern:
        - Install `hear21passt` via pip.
        - Use `hear21passt.base.load_model` and `get_timestamp_embeddings` /
          `get_scene_embeddings` to obtain per-frame embeddings.
        - Reshape to (B, F, T) so that T is the time axis used by our FFT.
        - Return embeddings and corresponding labels.
    """
    raise NotImplementedError(
        "get_passt_embeddings is a skeleton. Implement using hear21passt "
        "and return (embeddings, labels)."
    )


def get_panns_embeddings(
    file_list: Sequence[str],
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for PANN-S embedding extraction.

    Recommended pattern:
        - Prefer using a stable torch hub / pip interface if available.
        - If necessary, vendor only the minimal model definition into an
          `external/panns/` folder and load checkpoints from there.
        - Produce embeddings of shape (B, F, T) compatible with our FFT axis.
    """
    raise NotImplementedError(
        "get_panns_embeddings is a skeleton. Implement using official PANN-S "
        "models or a minimal vendored copy."
    )


def get_beats_embeddings(
    file_list: Sequence[str],
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for BEATs embedding extraction via SpeechBrain.

    Recommended pattern:
        - Install `speechbrain`.
        - Use `speechbrain.lobes.models.beats` and the official BEATs
          checkpoints to extract frame-level embeddings.
        - Reshape to (B, F, T).
    """
    raise NotImplementedError(
        "get_beats_embeddings is a skeleton. Implement using SpeechBrain's BEATs."
    )


def extract_and_save_embeddings(
    model: ModelName,
    dataset: str,
    file_list: Sequence[str],
    labels: Sequence[str],
    out_path: str,
) -> None:
    """
    High-level entry point: extract embeddings for a given (model, dataset)
    and save them into a standard NPZ format.

    Saved NPZ keys:
        - 'embeddings': np.ndarray, shape (B, F, T)
        - 'labels':     np.ndarray[str], shape (B,)
        - 'file_paths': np.ndarray[str], shape (B,)
    """
    if len(file_list) == 0:
        raise ValueError("file_list is empty.")
    if len(file_list) != len(labels):
        raise ValueError(
            f"file_list and labels must have same length "
            f"({len(file_list)} vs {len(labels)})."
        )

    if model == "clap":
        embeddings, y = get_clap_embeddings(file_list, dataset)
    elif model == "passt":
        embeddings, y = get_passt_embeddings(file_list, dataset)
    elif model == "pann_s":
        embeddings, y = get_panns_embeddings(file_list, dataset)
    elif model == "beats":
        embeddings, y = get_beats_embeddings(file_list, dataset)
    else:
        raise ValueError(f"Unknown model '{model}'.")

    if embeddings.shape[0] != len(file_list):
        raise ValueError(
            f"embeddings batch size {embeddings.shape[0]} "
            f"does not match file_list length {len(file_list)}."
        )

    # Convert labels and file paths to numpy arrays for saving.
    y_arr = np.asarray(y)
    file_arr = np.asarray(file_list)

    np.savez(
        out_path,
        embeddings=embeddings,
        labels=y_arr,
        file_paths=file_arr,
    )

