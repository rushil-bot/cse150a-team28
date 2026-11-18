"""Watch the agent drive the roundabout in a window.

Each step prints the state the agent is in and the action it took, so a crash
can be explained and not just seen. Random actions for now; once a learner
exists, swap `random_policy` for its trained policy.

    python scripts/watch.py            # 3 episodes
    python scripts/watch.py 10         # 10 episodes
"""

import sys

from av_gap.discretize import describe, encode
from av_gap.envs import make_env
from av_gap.rollout import random_policy


def main(episodes: int = 3, render_mode: str = "human") -> None:
    # real_time_rendering slows playback to wall-clock speed so it is watchable.
    env = make_env(render_mode=render_mode)
    env.unwrapped.configure({"real_time_rendering": True})
    action_names = env.unwrapped.action_type.actions
    policy = random_policy(env)

    for seed in range(episodes):
        print(f"seed {seed}")
        obs, _info = env.reset(seed=seed)
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy(obs)
            print(f"  {step:2}  {describe(encode(obs)):<52}  {action_names[action]}")
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            step += 1
        outcome = "crashed" if terminated else "survived"
        print(f"  {outcome} after {step} decisions, return {total_reward:.2f}\n")

    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
