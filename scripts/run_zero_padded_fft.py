import argparse
import os
from typing import List

from src.analysis.zero_padded_fft import analyze_zero_padding_for_dataset


def _parse_factors(values: List[str]) -> List[int]:
    return [int(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-padded FFT sanity checks for CLAP embeddings."
    )
    parser.add_argument(
        "--config_names",
        nargs="+",
        default=["fma_main", "asap_composer"],
        help="Embedding config names, e.g. fma_main asap_composer.",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=["1", "2", "4"],
        help=(
            "Padding factors, e.g. 1 2 4. "
            "For factor=1 we keep the original length (no zero padding). "
            "For factor>1 we zero-pad at the end so that T_pad = factor * T."
        ),
    )
    parser.add_argument(
        "--window_type",
        type=str,
        default="none",
        choices=["none", "hann", "hamming", "blackman"],
        help="FFT window type; 'none' means rectangular (no window).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help=(
            "Maximum number of samples per dataset used for zero-padding analysis. "
            "If the dataset is larger, a random subset of this size is used to "
            "avoid running out of memory."
        ),
    )
    args = parser.parse_args()

    factors = _parse_factors(args.factors)
    window = None if args.window_type == "none" else args.window_type

    for cfg_name in args.config_names:
        # Basic out_dir per config
        out_dir = os.path.join("results", "zero_padded_fft", cfg_name)

        # Use all 8 FMA styles / ASAP composers as subsets to visualise.
        subset_labels: List[str] = []
        if cfg_name == "fma_main":
            subset_labels = [
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
            subset_labels = ["Bach", "Beethoven", "Schubert", "Chopin", "Liszt"]

        analyze_zero_padding_for_dataset(
            config_name=cfg_name,
            factors=factors,
            window_type=window,
            out_dir=out_dir,
            max_samples=args.max_samples,
            subset_labels=subset_labels,
        )


if __name__ == "__main__":
    main()

