"""Environment factory.

The roundabout is the project's main scenario. The highway is kept so it can
be added later if there is time; nothing in the current plan depends on it.

Both scenarios use HighwayEnv's discrete maneuver actions:
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER.

`roundabout-v1` is the connected-lane roundabout. `roundabout-v0` is the
same map, but background vehicles miss neighbors across road segments and
crash more often than they should.
"""

from __future__ import annotations

import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

SCENARIOS = {
    "roundabout": "roundabout-v1",
    "highway": "highway-v0",
}

# HighwayEnv config keys corresponding to the proposal reward terms.
# Vary collision_reward against high_speed_reward for the frozen-robot ablation.
# The roundabout ignores right_lane_reward by default (its weight is 0).
REWARD_KEYS = (
    "collision_reward",
    "high_speed_reward",
    "right_lane_reward",
    "lane_change_reward",
)


def make_env(
    scenario: str = "roundabout",
    reward: dict | None = None,
    render_mode: str | None = None,
):
    """Build one scenario.

    `reward` overrides HighwayEnv's reward weights, for example
    `{"collision_reward": -1, "high_speed_reward": 0.4}`.
    """
    if scenario not in SCENARIOS:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"Unknown scenario {scenario!r}. Expected one of: {known}")

    config = None
    if reward:
        unknown = set(reward) - set(REWARD_KEYS)
        if unknown:
            raise KeyError(f"Unknown reward keys: {sorted(unknown)}")
        config = dict(reward)

    return gym.make(SCENARIOS[scenario], config=config, render_mode=render_mode)
