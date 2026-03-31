"""
Run multi-layer FFT spectral profiling for selected genres.

Assumes you have already extracted multi-layer embeddings with
`extract_embeddings_multi_layer` and saved them to
`data_artifacts/clap_embeddings_t64_multilayer.npz`.
"""
import argparse

from src.analysis.multi_layer_spectral import (
    MULTI_EMBEDDINGS_PATH,
    compute_multi_layer_spectral_profiles,
)


def main():
    parser = argparse.ArgumentParser(description="Run multi-layer spectral profiling.")
    parser.add_argument(
        "--npz_path",
        type=str,
        default=MULTI_EMBEDDINGS_PATH,
        help="Path to multi-layer embeddings npz (default: clap_embeddings_t64_multilayer.npz).",
    )
    parser.add_argument(
        "--genres",
        nargs="*",
        default=["Rock", "Pop", "Electronic", "Experimental"],
        help="Target genres for multi-layer spectral profiling.",
    )
    args = parser.parse_args()

    compute_multi_layer_spectral_profiles(args.genres, path=args.npz_path)


if __name__ == "__main__":
    main()

