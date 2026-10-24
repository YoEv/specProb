# Spectral Probing of Frozen Audio Encoders

**An Exploratory Study of Low-Frequency Concentration**  
Xiaosha Li and Alexander Lerch · Georgia Institute of Technology

[Technical report](SPECTRAL%20PROBING%20OF%20FROZEN%20AUDIO%20ENCODERS%3A%20AN%20EXPLORATORY%20STUDY%20OF%20LOW-FREQUENCY%20CONCENTRATIO.pdf)

## Overview

This repository investigates how task-relevant information is distributed across the temporal-frequency components of frozen audio representations. Rather than analyzing the acoustic spectrum of a waveform, we apply spectral transforms along the **time axis of encoder embeddings** and evaluate the information recoverable from individual frequency bands.

The study examines low-frequency concentration across frozen audio encoders and music information retrieval tasks. Its central question is whether task timescale predicts where discriminative information appears in the representation spectrum, or whether low-frequency components remain informative across tasks.

## Method

1. Extract temporally resolved features from a frozen audio encoder.
2. Apply FFT or DCT along the temporal dimension.
3. Construct individual-band, cumulative-band, and full-spectrum features.
4. Fit linear classification probes and compare predictive performance and spectral weight profiles.
5. Use control experiments to distinguish representation structure from effects of temporal ordering, DC components, labels, and model initialization.

The research spans CLAP, AudioMAE, BEATs, and PaSST. This public snapshot contains extraction and analysis utilities for several of these settings; it is a research codebase rather than a packaged reproduction of every experiment in the report. Refer to the report for the evaluated configurations and interpretation of results.

## Repository structure

| Directory | Contents |
| --- | --- |
| `src/analysis/` | Spectral transforms, statistics, and diagnostic analyses |
| `src/features/` | Feature extraction and representation preparation |
| `src/training/` | Linear probing and classification utilities |
| `src/visualization/` | Spectral profiles and diagnostic figures |
| `scripts/` | Feature extraction and experiment entry points |
| `tests/` | Sanity-check tests |
| `reports/`, `results/` | Experiment notes and analysis summaries |
| `projPlan/` | Retained English experiment plans; some describe exploratory work |
| `spectral-probing/`, `third_party/` | External implementations and encoder support; retain upstream attribution |

## Environment and data

The recorded research environment is provided in `requirements.runtime.txt`. It includes GPU-specific packages and is not a portable minimal installation specification. Adapt the PyTorch/CUDA installation to your hardware and use a separate virtual environment.

Dataset audio, pretrained checkpoints, and extracted embeddings must be obtained separately. Extraction scripts may require local path configuration. Obtain datasets and model weights under their original terms; no rights to those resources are granted by this repository.

For generic band probing, prepare an NPZ file containing `embeddings` with shape `(B, F, T)` and a label array such as `genres`. Here, `B` is the number of examples, `F` the feature dimension, and `T` the number of temporal positions.

```bash
python scripts/run_generic_fft_band_probe.py \
  --npz_path /path/to/embeddings.npz \
  --label_key genres \
  --target_label Rock \
  --transform_type fft \
  --band_width 4 \
  --results_dir results/example \
  --prefix example
```

This entry point performs a one-versus-rest analysis for the selected label and writes metrics and figures. It is an illustrative analysis command, not a claim of complete report reproduction.

## Interpretation and limitations

Temporal-frequency components of embeddings are not acoustic frequency bands. Probe performance measures recoverable information under a specified readout and split; it does not establish causal use of that information by the encoder. Low-frequency concentration should be interpreted with the stated controls and task configurations. Historical plans and exploratory summaries should not be treated as additional validated findings.

## Citation

If you use this work, please cite the technical report:

```bibtex
@techreport{li2026spectralprobing,
  title = {Spectral Probing of Frozen Audio Encoders: An Exploratory Study of Low-Frequency Concentration},
  author = {Li, Xiaosha and Lerch, Alexander},
  institution = {Georgia Institute of Technology},
  year = {2026},
  type = {Technical report},
  url = {https://github.com/YoEv/specProb}
}
```

## Acknowledgments

This repository includes external spectral-probing code and BEATs support. Please preserve the notices distributed with those components and cite the corresponding encoder and method papers when using them.
