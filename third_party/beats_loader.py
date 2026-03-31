"""
Utility for loading a BEATs model checkpoint from the official Microsoft implementation.

Requirements:
    - The official BEATs code (at least BEATs.py and backbone.py) is available under
      `third_party/beats/`, with a module layout compatible with:

          from beats.BEATs import BEATs, BEATsConfig

    - A downloaded BEATs checkpoint file (e.g. BEATs_iter3 or BEATs_iter3+ AS2M),
      stored somewhere inside the project (we assume under `data_artifacts/`).
"""

from __future__ import annotations

from typing import Union

import os
import sys
import torch

# Ensure that local BEATs sources are importable as `beats`, even if they live in
# nested folders like `third_party/beats/beats/BEATs.py` (as in the official repo).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Candidate roots where a `beats` *package* might live.
# In your current layout (`third_party/beats/beats/BEATs.py`), the package
# directory is the inner `third_party/beats/beats`, so we need its *parent*
# (`third_party/beats`) on sys.path, not `third_party` itself.
_CANDIDATES = [
    os.path.join(_THIS_DIR, "beats"),  # e.g. third_party/beats  -> contains `beats/BEATs.py`
]
for _p in _CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

try:
    from beats.BEATs import BEATs, BEATsConfig  # type: ignore[import]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import 'beats.BEATs'. Please ensure that the official "
        "BEATs code is available under 'third_party/beats/' or "
        "'third_party/beats/beats/' so that 'beats/BEATs.py' can be imported."
    ) from exc


def load_beats_model(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cpu",
) -> BEATs:
    """
    Load a BEATs model from a checkpoint file.

    Args:
        checkpoint_path: Path to the downloaded BEATs *.pt checkpoint.
        device: Device string or torch.device, e.g. "cuda" or "cpu".

    Returns:
        A BEATs model moved to the requested device and set to eval() mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model

