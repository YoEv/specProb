from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """
    Configuration for a family of spectral experiments.

    This keeps CLAP embedding shape, FFT length and output file paths
    in one place so that scripts do not hard-code shapes.
    """

    name: str
    # CLAP last_hidden_state base shape before any custom reshape.
    # For current use-cases this is always (B, 768, 2, T_frames).
    n_channels: int  # 768
    n_segments: int  # 2
    frames_per_segment: int  # 32 or 64
    # FFT configuration on the time axis
    n_fft: int  # input length to rfft (32 or 64)
    n_coeffs: int  # rfft output length (17 or 33)
    # Path to NPZ file that stores embeddings for this experiment.
    embeddings_file: str

    @property
    def feature_dim(self) -> int:
        """Total feature dimension after collapsing (channels × segments)."""
        return self.n_channels * self.n_segments


FMA_MAIN = EmbeddingConfig(
    name="fma_main",
    n_channels=768,
    n_segments=2,
    frames_per_segment=64,
    n_fft=64,
    n_coeffs=33,
    embeddings_file="data_artifacts/clap_embeddings_t64.npz",
)

ASAP_COMPOSER = EmbeddingConfig(
    name="asap_composer",
    n_channels=768,
    n_segments=2,
    frames_per_segment=32,
    n_fft=32,
    n_coeffs=17,
    embeddings_file="data_artifacts/clap_embeddings_asap_t32.npz",
)


CONFIGS = {
    FMA_MAIN.name: FMA_MAIN,
    ASAP_COMPOSER.name: ASAP_COMPOSER,
}


def get_embedding_config(name: str) -> EmbeddingConfig:
    """
    Fetch an embedding configuration by name.

    Known values:
        - "fma_main": main FMA genre experiments, (B, 768, 2, 64) → (B, 1536, 64), n_coeffs=33.
        - "asap_composer": D_asap_100 composer task, (B, 768, 2, 32) → (B, 1536, 32), n_coeffs=17.
    """
    try:
        return CONFIGS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown embedding config '{name}'. Available: {list(CONFIGS)}") from exc

