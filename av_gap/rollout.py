"""Episode runner.

A policy is a function from the kinematics observation to a discrete action.
Model-based and Q-learning policies should use this same runner so crash rate
and return are computed the same way. Count `length` toward the shared
sample budget: every environment step counts once, for both methods.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Policy = Callable[[np.ndarray], int]


def random_policy(env) -> Policy:
    def policy(obs: np.ndarray) -> int:
        del obs
        return int(env.action_space.sample())

    return policy


def run_episode(env, policy: Policy, seed: int) -> dict:
    """Run one episode. `terminated` means the ego vehicle crashed."""
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    length = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        total_reward += float(reward)
        length += 1
    return {
        "return": total_reward,
        "length": length,
        "crashed": bool(terminated),
    }
