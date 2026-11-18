"""Turn a roundabout observation into a state index for the tabular learners.

The state is four binned quantities combined into one integer: how fast the
agent is going, where it is in the roundabout, how far away the most dangerous
car is, and how fast that gap is closing.
"""

from __future__ import annotations

import numpy as np

from av_gap.envs import COL, OBS_FEATURES

# Bin edges, tuned with scripts/bin_coverage.py so no single bin swallows most
# of the steps. Retune these rather than the logic below.

# Agent speed [m/s]. The roundabout switches between target speeds 0, 8, and 16.
SPEED_EDGES = (4.0, 12.0)

# Distance from the map origin [m] within which the agent is on the ring.
# The two circulating lanes sit at radius 20 and 24.
RING_RADIUS = 26.0

# Distance to the most dangerous car [m]. Bin 0 is reserved for "no car closing".
GAP_EDGES = (10.0, 20.0, 40.0)

# How fast that gap is shrinking [m/s].
CLOSING_EDGES = (3.0, 12.0)

APPROACHING, ON_RING, EXITING = 0, 1, 2

N_SPEED = len(SPEED_EDGES) + 1
N_PHASE = 3
N_GAP = len(GAP_EDGES) + 2  # one bin per interval, plus "no car closing"
N_CLOSING = len(CLOSING_EDGES) + 1

DIMS = (N_SPEED, N_PHASE, N_GAP, N_CLOSING)
n_states = int(np.prod(DIMS))

SPEED_NAMES = ("stopped", "medium", "fast")
PHASE_NAMES = ("approaching", "on ring", "exiting")
GAP_NAMES = ("no car closing", "<10 m", "10-20 m", "20-40 m", ">=40 m")
CLOSING_NAMES = ("slow", "medium", "fast")


def _speed_bin(ego: np.ndarray) -> int:
    speed = float(np.hypot(ego[COL["vx"]], ego[COL["vy"]]))
    return int(np.digitize(speed, SPEED_EDGES))


def _phase_bin(ego: np.ndarray) -> int:
    """Where the agent is. Row 0 is in map coordinates, centered on the ring."""
    position = ego[[COL["x"], COL["y"]]]
    velocity = ego[[COL["vx"], COL["vy"]]]
    if float(np.hypot(*position)) <= RING_RADIUS:
        return ON_RING
    # Outside the ring, moving inward means approaching and outward means leaving.
    return APPROACHING if float(position @ velocity) < 0 else EXITING


def nearest_threat(obs: np.ndarray) -> tuple[float, float, float] | None:
    """The neighbor we would reach soonest, as (distance, closing speed, seconds).

    Neighbors whose gap is growing are ignored, as are empty slots. Returns None
    when nothing is closing in.
    """
    best = None
    for row in obs[1:]:
        if row[COL["presence"]] == 0:
            continue
        offset = row[[COL["x"], COL["y"]]]
        relative_velocity = row[[COL["vx"], COL["vy"]]]
        distance = float(np.hypot(*offset))
        if distance < 1e-6:
            continue
        closing = -float(offset @ relative_velocity) / distance
        if closing <= 0:
            continue
        seconds = distance / closing
        if best is None or seconds < best[2]:
            best = (distance, closing, seconds)
    return best


def encode(obs: np.ndarray) -> int:
    """Map one observation from `make_env` to a state index in [0, n_states)."""
    if obs.ndim != 2 or obs.shape[1] != len(OBS_FEATURES):
        raise ValueError(
            f"Expected an observation with {len(OBS_FEATURES)} columns, "
            f"got shape {obs.shape}"
        )
    ego = obs[0]
    threat = nearest_threat(obs)
    if threat is None:
        gap, closing = 0, 0
    else:
        distance, closing_speed, _seconds = threat
        gap = 1 + int(np.digitize(distance, GAP_EDGES))
        closing = int(np.digitize(closing_speed, CLOSING_EDGES))
    return int(
        np.ravel_multi_index((_speed_bin(ego), _phase_bin(ego), gap, closing), DIMS)
    )


def describe(state: int) -> str:
    """Readable form of a state index, for debugging and for the report."""
    speed, phase, gap, closing = np.unravel_index(state, DIMS)
    if gap == 0:
        return f"{SPEED_NAMES[speed]}, {PHASE_NAMES[phase]}, no car closing"
    return (
        f"{SPEED_NAMES[speed]}, {PHASE_NAMES[phase]}, "
        f"gap {GAP_NAMES[gap]}, closing {CLOSING_NAMES[closing]}"
    )