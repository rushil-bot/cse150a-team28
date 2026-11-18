"""Checks on the environment factory and the observation format.

These pin down the format every learner depends on. If one fails, the bins in
`av_gap/discretize.py` and both learners are reading something they did not
expect.
"""

import pytest

from av_gap.envs import COL, OBS_FEATURES, SCENARIOS, make_env


@pytest.fixture(scope="module")
def env():
    env = make_env()
    yield env
    env.close()


def test_default_scenario_is_the_roundabout():
    assert SCENARIOS["roundabout"] == "roundabout-v1"


def test_observation_shape(env):
    obs, _info = env.reset(seed=0)
    assert obs.shape == (5, len(OBS_FEATURES))


def test_five_maneuver_actions(env):
    env.reset(seed=0)
    assert env.unwrapped.action_type.actions == {
        0: "LANE_LEFT",
        1: "IDLE",
        2: "LANE_RIGHT",
        3: "FASTER",
        4: "SLOWER",
    }


def test_agent_row_is_present_and_has_a_heading(env):
    obs, _info = env.reset(seed=0)
    agent = obs[0]
    assert agent[COL["presence"]] == 1
    # cos_h and sin_h are a unit vector, so their squares sum to 1.
    heading = agent[COL["cos_h"]] ** 2 + agent[COL["sin_h"]] ** 2
    assert heading == pytest.approx(1.0, abs=1e-3)


def test_values_are_in_real_units_not_normalized(env):
    """Normalized observations are clipped to [-1, 1]; ours are meters and m/s."""
    obs, _info = env.reset(seed=0)
    positions = obs[:, [COL["x"], COL["y"]]]
    assert abs(positions).max() > 1.0


def test_reward_override_reaches_the_simulator():
    env = make_env(reward={"collision_reward": -5})
    try:
        assert env.unwrapped.config["collision_reward"] == -5
    finally:
        env.close()


def test_reward_override_keeps_the_observation_config():
    env = make_env(reward={"collision_reward": -5})
    try:
        obs, _info = env.reset(seed=0)
        assert obs.shape == (5, len(OBS_FEATURES))
    finally:
        env.close()


def test_misspelled_reward_key_is_rejected():
    with pytest.raises(KeyError):
        make_env(reward={"colision_reward": -5})


def test_unknown_scenario_is_rejected():
    with pytest.raises(KeyError):
        make_env("roundabout-v0")
