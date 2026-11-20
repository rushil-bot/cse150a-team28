"""Episode runner and baseline drivers.

A policy is a function from the kinematics observation to a discrete action.
Model-based and Q-learning policies should use this same runner so crash rate,
return, and time-to-collision are computed the same way. Count `length` toward
the shared sample budget: every environment step counts once, for both methods.

Two baselines live here:

- `random_policy` picks a random action every step.
- `IDMDriven` hands the agent's car to HighwayEnv's IDM driver, the same
  car-following and lane-change rules the background cars use. Run it with
  `idm_policy`, whose action the IDM car ignores.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

from av_gap.discretize import nearest_threat

Policy = Callable[[np.ndarray], int]

IDLE = 1


def random_policy(env) -> Policy:
    def policy(obs: np.ndarray) -> int:
        del obs
        return int(env.action_space.sample())

    return policy


def time_to_collision(obs: np.ndarray) -> float:
    """Seconds until the agent reaches the soonest-contact neighbor.

    Assumes nobody changes speed. Uses `nearest_threat`, so any closing neighbor
    counts, not only the car ahead. Returns infinity when nothing is closing in.
    """
    threat = nearest_threat(obs)
    return math.inf if threat is None else threat[2]


def run_episode(env, policy: Policy, seed: int) -> dict:
    """Run one episode. `terminated` means the ego vehicle crashed.

    `min_ttc` is the smallest time-to-collision seen over the episode,
    including the first observation, and is infinity if nothing ever closed in.
    """
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    min_ttc = time_to_collision(obs)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        total_reward += float(reward)
        length += 1
        min_ttc = min(min_ttc, time_to_collision(obs))
    return {
        "return": total_reward,
        "length": length,
        "crashed": bool(terminated),
        "min_ttc": min_ttc,
    }


class IDMDriven(gym.Wrapper):
    """Let HighwayEnv's IDM driver steer the agent's car.

    After every reset, the agent's car is replaced by an `IDMVehicle` with the
    same position, speed, target speed, and route. `IDMVehicle.act` ignores the
    action it is given, so the action passed to `step` changes nothing except
    the lane-change reward term. Pass `idm_policy` to keep that term at zero.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        agent = base.vehicle
        driver = IDMVehicle.create_from(agent)
        vehicles = base.road.vehicles
        vehicles[vehicles.index(agent)] = driver
        base.vehicle = driver
        return obs, info


def idm_policy(obs: np.ndarray) -> int:
    """The action to pass alongside `IDMDriven`. The IDM car ignores it."""
    del obs
    return IDLE
