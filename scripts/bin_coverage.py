"""Report how the state bins get used, to tune the edges in discretize.py.

    python scripts/bin_coverage.py        # 150 episodes
    python scripts/bin_coverage.py 400
"""

import sys
from collections import Counter

import numpy as np

from av_gap.discretize import DIMS, describe, encode, n_states
from av_gap.envs import make_env
from av_gap.rollout import random_policy


def main(episodes: int = 150) -> None:
    env = make_env()
    policy = random_policy(env)
    seen: Counter[int] = Counter()
    per_quantity = [np.zeros(size, int) for size in DIMS]
    steps = 0

    for seed in range(episodes):
        obs, _info = env.reset(seed=seed)
        done = False
        while not done:
            state = encode(obs)
            seen[state] += 1
            for axis, value in enumerate(np.unravel_index(state, DIMS)):
                per_quantity[axis][value] += 1
            obs, _reward, terminated, truncated, _info = env.step(policy(obs))
            done = terminated or truncated
            steps += 1
    env.close()

    print(f"{episodes} episodes, {steps} steps")
    print(f"visited {len(seen)} of {n_states} states")
    for name, counts in zip(("speed", "phase", "gap", "closing"), per_quantity):
        largest = counts.max() / max(counts.sum(), 1)
        print(f"  {name:8} {counts.tolist()}  largest bin {largest:.0%}")
    print("most common states:")
    for state, count in seen.most_common(5):
        print(f"  {count:5}  {describe(state)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)