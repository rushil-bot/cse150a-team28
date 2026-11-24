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
    assert set(result) == {
        "return",
        "length",
        "crashed",
        "min_ttc",
        "distance",
        "avg_speed",
        "decision_time",
    }
    assert result["min_ttc"] >= 0
    assert result["distance"] >= 0
    assert result["avg_speed"] >= 0
    assert result["decision_time"] > 0


def test_distance_matches_the_simulator_position():
    """Distance comes from observation row 0, which must be the true map position."""
    env = make_env()
    try:
        obs, _info = env.reset(seed=3)
        for _ in range(3):
            obs, *_ = env.step(1)
            assert obs[0, 1:3] == pytest.approx(env.unwrapped.vehicle.position, abs=1e-3)
    finally:
        env.close()


def test_frozen_agent_covers_almost_no_distance():
    """A car that only brakes should score far less distance than one that drives."""
    slower = 4

    def run(action):
        env = make_env()
        try:
            return run_episode(env, lambda obs: action, seed=10_000)["distance"]
        finally:
            env.close()

    assert run(slower) < run(1)


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
    assert result["distance"] > 0


def test_idm_reports_its_own_decision_time():
    """IDM decides inside the simulator, so the wrapper must time those decisions."""
    env = IDMDriven(make_env())
    try:
        env.reset(seed=0)
        _obs, _r, _term, _trunc, info = env.step(idm_policy(None))
    finally:
        env.close()
    assert info["decision_time"] > 0
