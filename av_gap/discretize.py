"""Turn a roundabout observation into a state index for the tabular learners.

Placeholder: every observation maps to state 0. The real bins replace this
without changing the names or signatures below.
"""

from __future__ import annotations

import numpy as np

OBS_SHAPE = (5, 7)

n_states = 1


def encode(obs: np.ndarray) -> int:
    """Map one observation from `make_env` to a state index in [0, n_states)."""
    if obs.shape != OBS_SHAPE:
        raise ValueError(f"Expected observation shape {OBS_SHAPE}, got {obs.shape}")
    return 0