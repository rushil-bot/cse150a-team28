"""Checks on the episode runner, the time-to-collision measure, and the IDM baseline."""

import math

import numpy as np
import pytest
from highway_env.vehicle.behavior import IDMVehicle

from av_gap.envs import OBS_FEATURES, make_env
from av_gap.rollout import (
    IDMDriven,
    idm_policy,
    random_policy,
    run_episode,
    time_to_collision,
)

# presence, x, y, vx, vy, cos_h, sin_h
APPROACHING_EGO = [1, 2, 45, 0, -8, 0, -1]


def build_obs(ego, neighbors=()):
    obs = np.zeros((5, len(OBS_FEATURES)), dtype=np.float32)
    for i, row in enumerate([ego, *neighbors]):
        obs[i] = row
    return obs


def test_empty_road_has_infinite_ttc():
    assert math.isinf(time_to_collision(build_obs(APPROACHING_EGO)))


def test_closing_car_has_smaller_ttc_than_empty_road():
    closing = [1, 0, -15, 0, 5, 0, -1]  # 15 m ahead, closing at 5 m/s
    ttc = time_to_collision(build_obs(APPROACHING_EGO, [closing]))
    assert ttc == pytest.approx(3, abs=0.1)
    assert ttc < time_to_collision(build_obs(APPROACHING_EGO))


def test_run_episode_reports_min_ttc():
    env = make_env()
    try:
        result = run_episode(env, random_policy(env), seed=0)
    finally:
        env.close()
    assert set(result) == {"return", "length", "crashed", "min_ttc"}
    assert result["min_ttc"] >= 0


def test_idm_wrapper_hands_the_agent_to_idm():
    env = IDMDriven(make_env())
    try:
        env.reset(seed=0)
        base = env.unwrapped
        assert isinstance(base.vehicle, IDMVehicle)
        assert base.vehicle in base.road.vehicles
        # Exactly one agent car on the road, not the old one plus the new one.
        assert sum(v is base.vehicle for v in base.road.vehicles) == 1
    finally:
        env.close()


def test_idm_baseline_runs_a_full_episode():
    env = IDMDriven(make_env())
    try:
        result = run_episode(env, idm_policy, seed=0)
    finally:
        env.close()
    assert result["length"] > 0
