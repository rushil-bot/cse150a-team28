"""Random-action baseline on the roundabout. Replace `random_policy` with a learned policy later."""

from av_gap.envs import make_env
from av_gap.rollout import random_policy, run_episode

SCENARIO = "roundabout"
SEEDS = range(5)


def main() -> None:
    env = make_env(SCENARIO)
    policy = random_policy(env)
    rows = [run_episode(env, policy, seed) for seed in SEEDS]
    env.close()
    mean_return = sum(row["return"] for row in rows) / len(rows)
    crash_rate = sum(row["crashed"] for row in rows) / len(rows)
    print(
        f"{SCENARIO}: mean return {mean_return:.3f}, "
        f"crash rate {crash_rate:.0%} over {len(rows)} seeds"
    )


if __name__ == "__main__":
    main()
