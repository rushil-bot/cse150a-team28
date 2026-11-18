"""Confirm the roundabout resets and steps. No learned policy."""

from av_gap.envs import SCENARIOS, make_env

SCENARIO = "roundabout"


def main() -> None:
    env = make_env(SCENARIO)
    obs, _info = env.reset(seed=0)
    print(SCENARIO, SCENARIOS[SCENARIO])
    print("  obs shape:", obs.shape)
    print("  agent row (map coords):", obs[0].round(2))
    print("  nearest neighbor (relative):", obs[1].round(2))
    print("  actions:", env.unwrapped.action_type.actions)
    action = env.action_space.sample()
    _obs, reward, terminated, truncated, _info = env.step(action)
    print(
        "  reward:",
        reward,
        "terminated:",
        terminated,
        "truncated:",
        truncated,
    )
    env.close()


if __name__ == "__main__":
    main()
