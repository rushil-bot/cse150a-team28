"""Checks on `encode` and the bins behind it.

The first few cover the contract the learners rely on: a valid index, the same
answer for the same observation, and a clear error on the wrong shape. The rest
use hand-built observations to pin down what each bin means.
"""

import numpy as np
import pytest

from av_gap.discretize import (
    DIMS,
    GAP_NAMES,
    describe,
    encode,
    n_states,
    nearest_threat,
)
from av_gap.envs import OBS_FEATURES, make_env
from av_gap.rollout import random_policy


def test_encode_returns_a_valid_state_index():
    env = make_env()
    try:
        obs, _info = env.reset(seed=0)
        state = encode(obs)
        assert isinstance(state, int)
        assert 0 <= state < n_states
    finally:
        env.close()


def test_encode_is_deterministic():
    env = make_env()
    try:
        obs, _info = env.reset(seed=0)
        assert encode(obs) == encode(obs)
    finally:
        env.close()


def test_every_step_of_an_episode_is_in_range():
    """Walk a whole episode so unusual observations are covered too."""
    env = make_env()
    try:
        obs, _info = env.reset(seed=0)
        policy = random_policy(env)
        done = False
        while not done:
            assert 0 <= encode(obs) < n_states
            obs, _reward, terminated, truncated, _info = env.step(policy(obs))
            done = terminated or truncated
    finally:
        env.close()


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        encode(np.zeros((5, 5)))


def test_encode_accepts_what_make_env_returns():
    env = make_env()
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape[1] == len(OBS_FEATURES)
        assert 0 <= encode(obs) < n_states
    finally:
        env.close()


def build_obs(ego, neighbors=()):
    """Assemble an observation. Rows not given stay empty (presence 0)."""
    obs = np.zeros((5, len(OBS_FEATURES)), dtype=np.float32)
    for i, row in enumerate([ego, *neighbors]):
        obs[i] = row
    return obs


# presence, x, y, vx, vy, cos_h, sin_h
APPROACHING_EGO = [1, 2, 45, 0, -8, 0, -1]  # south approach, heading inward


def test_empty_road_has_no_threat():
    assert nearest_threat(build_obs(APPROACHING_EGO)) is None


def test_car_pulling_away_is_not_a_threat():
    ahead_and_leaving = [1, 0, -20, 0, -5, 0, -1]
    assert nearest_threat(build_obs(APPROACHING_EGO, [ahead_and_leaving])) is None


def test_car_closing_is_found_with_the_right_gap():
    closing = [1, 0, -15, 0, 5, 0, -1]  # 15 m ahead, gap shrinking at 5 m/s
    distance, closing_speed, seconds = nearest_threat(
        build_obs(APPROACHING_EGO, [closing])
    )
    assert distance == pytest.approx(15, abs=0.1)
    assert closing_speed == pytest.approx(5, abs=0.1)
    assert seconds == pytest.approx(3, abs=0.1)


def test_nearest_threat_skips_empty_slots():
    """An empty slot is all zeros, which must not read as a car at zero distance."""
    far_car = [1, 0, -30, 0, 5, 0, -1]
    obs = build_obs(APPROACHING_EGO, [far_car])  # rows 2-4 stay empty
    distance, _closing, _seconds = nearest_threat(obs)
    assert distance == pytest.approx(30, abs=0.1)


def test_phase_bins_differ_by_where_the_agent_is():
    on_ring = [1, 0, 22, -8, 0, -1, 0]
    exiting = [1, 2, 45, 0, 8, 0, 1]
    states = {
        encode(build_obs(APPROACHING_EGO)),
        encode(build_obs(on_ring)),
        encode(build_obs(exiting)),
    }
    assert len(states) == 3


def test_n_states_matches_the_bin_sizes():
    assert n_states == int(np.prod(DIMS))


def test_describe_reports_an_empty_road():
    state = encode(build_obs(APPROACHING_EGO))
    assert GAP_NAMES[0] in describe(state)


def test_a_short_run_visits_many_states():
    """Guards against bins collapsing to one or two values."""
    env = make_env()
    try:
        policy = random_policy(env)
        seen = set()
        for seed in range(30):
            obs, _info = env.reset(seed=seed)
            done = False
            while not done:
                seen.add(encode(obs))
                obs, _r, terminated, truncated, _info = env.step(policy(obs))
                done = terminated or truncated
    finally:
        env.close()
    assert len(seen) >= 20