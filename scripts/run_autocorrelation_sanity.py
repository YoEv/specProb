import argparse
import os
from typing import List

from src.analysis.autocorrelation_checks import analyze_autocorrelation_for_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run autocorrelation-based sanity checks for CLAP embeddings."
    )
    parser.add_argument(
        "--config_names",
        nargs="+",
        default=["fma_main", "asap_composer"],
        help="Embedding config names, e.g. fma_main asap_composer.",
    )
    parser.add_argument(
        "--max_lag_fma",
        type=int,
        default=64,
        help="Max lag for fma_main (will be clipped to T-1 if too large).",
    )
    parser.add_argument(
        "--max_lag_asap",
        type=int,
        default=32,
        help="Max lag for asap_composer (will be clipped to T-1 if too large).",
    )
    parser.add_argument(
        "--n_features_sample",
        type=int,
        default=128,
        help="Number of feature dims to sample per dataset for autocorr averaging.",
    )
    args = parser.parse_args()

    for cfg_name in args.config_names:
        if cfg_name == "fma_main":
            max_lag = args.max_lag_fma
            subset_labels: List[str] = [
                "Electronic",
                "Experimental",
                "Folk",
                "Hip-Hop",
                "Instrumental",
                "International",
                "Pop",
                "Rock",
            ]
        elif cfg_name == "asap_composer":
            max_lag = args.max_lag_asap
            subset_labels = ["Bach", "Beethoven", "Schubert", "Chopin", "Liszt"]
        else:
            # Default conservative max_lag; will be clipped inside if needed.
            max_lag = 32
            subset_labels = []

        out_dir = os.path.join("results", "autocorrelation", cfg_name)

        analyze_autocorrelation_for_dataset(
            config_name=cfg_name,
            max_lag=max_lag,
            out_dir=out_dir,
            subset_labels=subset_labels,
            n_features_sample=args.n_features_sample,
        )


if __name__ == "__main__":
    main()

