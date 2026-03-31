"""
Run BPM estimation and BPM vs spectral-peak energy correlation (Task 1, BPM sanity check).
"""
from src.analysis.bpm_correlation import (
    EMBEDDINGS_PATH,
    compute_bpm_energy_correlation,
    estimate_bpm_for_all_tracks,
)


def main():
    bpm_df = estimate_bpm_for_all_tracks()
    compute_bpm_energy_correlation(embeddings_path=EMBEDDINGS_PATH, bpm_df=bpm_df)


if __name__ == "__main__":
    main()

