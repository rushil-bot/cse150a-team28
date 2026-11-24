"""Episode runner and baseline drivers.

A policy is a function from the kinematics observation to a discrete action.
Model-based and Q-learning policies should use this same runner so every
metric is computed the same way. Count `length` toward the shared sample
budget: every environment step counts once, for both methods.

Two baselines live here:

- `random_policy` picks a random action every step.
- `IDMDriven` hands the agent's car to HighwayEnv's IDM driver, the same
  car-following and lane-change rules the background cars use. Run it with
  `idm_policy`, whose action the IDM car ignores.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

from av_gap.discretize import nearest_threat
from av_gap.envs import COL

Policy = Callable[[np.ndarray], int]

IDLE = 1

# An environment that makes the agent's decisions itself (IDMDriven) reports
# how long they took under this `info` key, in seconds per decision.
DECISION_TIME_KEY = "decision_time"


def random_policy(env, seed: int | None = None) -> Policy:
    """Pick a uniformly random action every step.

    Pass `seed` for repeatable actions. Seeding `env.action_space` does not
    work: HighwayEnv builds a fresh action space on every reset.
    """
    n_actions = int(env.action_space.n)
    rng = np.random.default_rng(seed)

    def policy(obs: np.ndarray) -> int:
        del obs
        return int(rng.integers(n_actions))

    return policy


def time_to_collision(obs: np.ndarray) -> float:
    """Seconds until the agent reaches the soonest-contact neighbor.

    Assumes nobody changes speed. Uses `nearest_threat`, so any closing neighbor
    counts, not only the car ahead. Returns infinity when nothing is closing in.
    """
    threat = nearest_threat(obs)
    return math.inf if threat is None else threat[2]


def _position(obs: np.ndarray) -> np.ndarray:
    """The agent's map position. Row 0 of the observation is in map coordinates."""
    return obs[0, [COL["x"], COL["y"]]].astype(float)


def _speed(obs: np.ndarray) -> float:
    return float(np.hypot(obs[0, COL["vx"]], obs[0, COL["vy"]]))


def run_episode(env, policy: Policy, seed: int) -> dict:
    """Run one episode and score it. `crashed` means the episode ended in a crash.

    Returns:
        return: summed reward.
        length: number of `env.step` calls (decisions).
        crashed: whether the agent crashed.
        min_ttc: smallest time-to-collision [s] over the episode, including the
            first observation; infinity if nothing ever closed in.
        distance: meters the agent covered, summed between decisions. Straight
            lines between snapshots, so curves are very slightly undercounted.
            A crash stops the count, so crashing early scores low.
        avg_speed: mean agent speed [m/s] over the observations after each step.
        decision_time: mean wall-clock seconds per decision, excluding
            simulation. For ordinary policies this times `policy(obs)`. If the
            environment decides for the agent (IDMDriven), it reports its own
            decision time in `info`, and that is used instead.
    """
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    min_ttc = time_to_collision(obs)
    distance = 0.0
    speed_sum = 0.0
    decision_sum = 0.0
    position = _position(obs)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        start = time.perf_counter()
        action = policy(obs)
        policy_seconds = time.perf_counter() - start
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        length += 1
        min_ttc = min(min_ttc, time_to_collision(obs))
        new_position = _position(obs)
        distance += float(np.linalg.norm(new_position - position))
        position = new_position
        speed_sum += _speed(obs)
        decision_sum += info.get(DECISION_TIME_KEY, policy_seconds)
    return {
        "return": total_reward,
        "length": length,
        "crashed": bool(terminated),
        "min_ttc": min_ttc,
        "distance": distance,
        "avg_speed": speed_sum / length,
        "decision_time": decision_sum / length,
    }


class _TimedIDMVehicle(IDMVehicle):
    """An IDM car that records how long each of its decisions takes."""

    def act(self, action=None):
        if self.crashed:
            # Nothing is decided after a crash; don't let the instant return
            # pull the average down.
            return super().act(action)
        start = time.perf_counter()
        super().act(action)
        self.decision_seconds.append(time.perf_counter() - start)


class IDMDriven(gym.Wrapper):
    """Let HighwayEnv's IDM driver steer the agent's car.

    After every reset, the agent's car is replaced by an `IDMVehicle` with the
    same position, speed, target speed, and route. `IDMVehicle.act` ignores the
    action it is given, so the action passed to `step` changes nothing except
    the lane-change reward term. Pass `idm_policy` to keep that term at zero.

    The IDM car re-decides its steering and acceleration on every simulation
    frame. `step` reports the mean time of those decisions under
    `info["decision_time"]`, so `run_episode` times IDM doing the same job as a
    learned policy: turning what it sees into a control choice.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        agent = base.vehicle
        driver = _TimedIDMVehicle.create_from(agent)
        driver.decision_seconds = []
        vehicles = base.road.vehicles
        vehicles[vehicles.index(agent)] = driver
        base.vehicle = driver
        return obs, info

    def step(self, action):
        driver = self.env.unwrapped.vehicle
        driver.decision_seconds = []
        obs, reward, terminated, truncated, info = self.env.step(action)
        if driver.decision_seconds:
            info[DECISION_TIME_KEY] = float(np.mean(driver.decision_seconds))
        return obs, reward, terminated, truncated, info


def idm_policy(obs: np.ndarray) -> int:
    """The action to pass alongside `IDMDriven`. The IDM car ignores it."""
    del obs
    return IDLE
