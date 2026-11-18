"""Watch the agent drive the roundabout in a window.

Random actions for now. Once a learner exists, swap `random_policy` for its
trained policy to see what it actually does.

    python scripts/watch.py            # 3 episodes
    python scripts/watch.py 10         # 10 episodes
"""

import sys

from av_gap.envs import make_env
from av_gap.rollout import random_policy, run_episode


def main(episodes: int = 3, render_mode: str = "human") -> None:
    # real_time_rendering slows playback to wall-clock speed so it is watchable.
    env = make_env(render_mode=render_mode)
    env.unwrapped.configure({"real_time_rendering": True})
    policy = random_policy(env)
    for seed in range(episodes):
        row = run_episode(env, policy, seed)
        outcome = "crashed" if row["crashed"] else "survived"
        print(
            f"seed {seed}: {outcome} after {row['length']} decisions, "
            f"return {row['return']:.2f}"
        )
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
