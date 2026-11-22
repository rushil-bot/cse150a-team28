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

# Rows: 0 is the agent's car, 1-4 are the nearest neighbors.
# Row 0 is in map coordinates. In rows 1-4, x, y, vx, vy are offsets from the
# agent's car, and cos_h, sin_h are that neighbor's own map heading.
# Units are meters and m/s.
OBS_FEATURES = ("presence", "x", "y", "vx", "vy", "cos_h", "sin_h")
OBS_CONFIG = {
    "type": "Kinematics",
    "vehicles_count": 5,
    "features": list(OBS_FEATURES),
    "absolute": False,
    "normalize": False,
    # The roundabout hides cars more than two car lengths behind along the agent's
    # lane, which can drop circulating cars approaching the entry.
    "see_behind": True,
    "order": "sorted",
}
COL = {name: i for i, name in enumerate(OBS_FEATURES)}


def make_env(
    scenario: str = "roundabout",
    reward: dict | None = None,
    render_mode: str | None = None,
    calibrated_traffic: bool = False,
):
    """Build one scenario.

    `reward` overrides HighwayEnv's reward weights, for example
    `{"collision_reward": -1, "high_speed_reward": 0.4}`.

    `calibrated_traffic=True` makes the roundabout's background cars use
    settings measured from rounD (see `av_gap/traffic.py`). The default keeps
    HighwayEnv's built-in drivers.
    """
    if scenario not in SCENARIOS:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"Unknown scenario {scenario!r}. Expected one of: {known}")

    config = {"observation": dict(OBS_CONFIG)}
    if reward:
        unknown = set(reward) - set(REWARD_KEYS)
        if unknown:
            raise KeyError(f"Unknown reward keys: {sorted(unknown)}")
        config.update(reward)

    if calibrated_traffic:
        if scenario != "roundabout":
            raise ValueError("calibrated_traffic is measured on roundabouts only")
        config["other_vehicles_type"] = "av_gap.traffic.CalibratedVehicle"

    return gym.make(SCENARIOS[scenario], config=config, render_mode=render_mode)

