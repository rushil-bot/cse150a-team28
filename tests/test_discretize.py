"""Checks on `encode`.

`encode` is still a placeholder that returns 0 for every observation, so these
only cover the contract the learners rely on: a valid index, the same answer
for the same observation, and a clear error on the wrong shape. Sub-issue 1.5
adds tests for the bins themselves.
"""

import numpy as np
import pytest

from av_gap.discretize import OBS_SHAPE, encode, n_states
from av_gap.envs import make_env
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


def test_obs_shape_matches_what_make_env_returns():
    env = make_env()
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == OBS_SHAPE
    finally:
        env.close()
