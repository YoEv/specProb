import argparse
import os
from typing import List

from src.analysis.multi_model_periodicity import (
    compare_autocorrelation_across_models,
    compare_fft_across_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model FFT + autocorrelation sanity checks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fma_main", "asap_composer"],
        help="Datasets to analyse, e.g. fma_main asap_composer.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clap"],
        help=(
            "Models to include, e.g. clap passt pann_s beats. "
            "By default only 'clap' is enabled; other models require "
            "precomputed NPZ embeddings and corresponding specs in "
            "src.config.multi_model_sanity."
        ),
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=["1", "2"],
        help="Zero-padding factors, e.g. 1 2 4.",
    )
    parser.add_argument(
        "--window_type",
        type=str,
        default="none",
        choices=["none", "hann", "hamming", "blackman"],
        help="FFT window type; 'none' means rectangular (no window).",
    )
    parser.add_argument(
        "--max_lag_fma",
        type=int,
        default=64,
        help="Max lag for fma_main autocorrelation.",
    )
    parser.add_argument(
        "--max_lag_asap",
        type=int,
        default=32,
        help="Max lag for asap_composer autocorrelation.",
    )
    parser.add_argument(
        "--n_features_sample",
        type=int,
        default=128,
        help=(
            "Number of feature dims sampled per model for autocorrelation. "
            "Set to 0 or a negative value to use all feature dimensions."
        ),
    )
    args = parser.parse_args()

    factors: List[int] = [int(v) for v in args.factors]
    window = None if args.window_type == "none" else args.window_type

    for dataset in args.datasets:
        out_dir = os.path.join("results", "multi_model_sanity", dataset)

        # FFT comparison
        compare_fft_across_models(
            dataset=dataset,
            models=args.models,
            factors=factors,
            window_type=window,
            out_dir=out_dir,
        )

        # Autocorrelation comparison
        if dataset == "fma_main":
            max_lag = args.max_lag_fma
        elif dataset == "asap_composer":
            max_lag = args.max_lag_asap
        else:
            max_lag = 32

        # If n_features_sample <= 0, use all feature dimensions (no subsampling).
        n_feat = None if args.n_features_sample <= 0 else args.n_features_sample

        compare_autocorrelation_across_models(
            dataset=dataset,
            models=args.models,
            max_lag=max_lag,
            out_dir=out_dir,
            n_features_sample=n_feat,
        )


if __name__ == "__main__":
    main()

