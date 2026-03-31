from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ModelEmbeddingSpec:
    """
    Specification for a (model, dataset) embedding tensor stored on disk.

    This decouples analysis code from how embeddings are actually extracted.
    """

    model: str          # 'clap' | 'passt' | 'pann_s' | 'beats'
    dataset: str        # 'fma_main' | 'asap_composer' | ...
    embeddings_file: str
    feature_dim: int    # F, e.g. 1536
    time_length: int    # T, e.g. 64 or 32


# Known specs. For now we only have CLAP; others are placeholders
# to be filled once corresponding extraction scripts are run.

MODEL_SPECS: Dict[Tuple[str, str], ModelEmbeddingSpec] = {
    # CLAP on FMA / LMD
    ("clap", "fma_main"): ModelEmbeddingSpec(
        model="clap",
        dataset="fma_main",
        embeddings_file="data_artifacts/clap_embeddings_t64.npz",
        feature_dim=1536,
        time_length=64,
    ),
    # CLAP on ASAP composer task
    ("clap", "asap_composer"): ModelEmbeddingSpec(
        model="clap",
        dataset="asap_composer",
        embeddings_file="data_artifacts/clap_embeddings_asap_t32.npz",
        feature_dim=1536,
        time_length=32,
    ),
    # PASST on FMA (spec derived from existing NPZ)
    ("passt", "fma_main"): ModelEmbeddingSpec(
        model="passt",
        dataset="fma_main",
        embeddings_file="data_artifacts/passt_embeddings_t64.npz",
        feature_dim=1295,
        time_length=601,
    ),
    # PASST on ASAP composer task (spec derived from existing NPZ)
    ("passt", "asap_composer"): ModelEmbeddingSpec(
        model="passt",
        dataset="asap_composer",
        embeddings_file="data_artifacts/passt_embeddings_asap_t32.npz",
        feature_dim=1295,
        time_length=201,
    ),
    ("pann_s", "fma_main"): ModelEmbeddingSpec(
        model="pann_s",
        dataset="fma_main",
        embeddings_file="data_artifacts/panns_embeddings_t64.npz",
        feature_dim=1536,
        time_length=64,
    ),
    # BEATs on FMA (spec from extracted NPZ: (800, 1, 527))
    ("beats", "fma_main"): ModelEmbeddingSpec(
        model="beats",
        dataset="fma_main",
        embeddings_file="data_artifacts/beats_embeddings_t64.npz",
        feature_dim=1,
        time_length=527,
    ),
    # BEATs on ASAP composer (spec from extracted NPZ: (100, 1, 527))
    ("beats", "asap_composer"): ModelEmbeddingSpec(
        model="beats",
        dataset="asap_composer",
        embeddings_file="data_artifacts/beats_embeddings_asap_t32.npz",
        feature_dim=1,
        time_length=527,
    ),
}


def get_model_spec(model: str, dataset: str) -> ModelEmbeddingSpec:
    """
    Fetch a ModelEmbeddingSpec by (model, dataset) key.

    Example:
        spec = get_model_spec("clap", "fma_main")
    """
    key = (model, dataset)
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        available = ", ".join(f"{m}:{d}" for (m, d) in MODEL_SPECS.keys())
        raise KeyError(
            f"Unknown (model, dataset) combination {key}. "
            f"Available specs: {available}"
        ) from exc

