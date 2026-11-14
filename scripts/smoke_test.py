"""Confirm both scenarios reset and step. No learned policy."""

from av_gap.envs import SCENARIOS, make_env


def main() -> None:
    for scenario in SCENARIOS:
        env = make_env(scenario)
        obs, _info = env.reset(seed=0)
        print(scenario, SCENARIOS[scenario])
        print("  obs shape:", obs.shape)
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
