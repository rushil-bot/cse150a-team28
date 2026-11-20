"""Score the baselines on the roundabout.

Runs every policy through `run_episode` on the same seeds and prints mean
return, crash rate, and minimum time-to-collision for each. Learned policies
from tasks 2 and 3 get added to `POLICIES` once their `train(...)` exists.

    python scripts/evaluate.py              # 20 episodes per policy
    python scripts/evaluate.py --episodes 100
"""

from __future__ import annotations

import argparse
import math
import statistics

from av_gap.envs import make_env
from av_gap.rollout import IDMDriven, idm_policy, random_policy, run_episode

SCENARIO = "roundabout"


def random_baseline():
    env = make_env(SCENARIO)
    return env, random_policy(env)


def idm_baseline():
    return IDMDriven(make_env(SCENARIO)), idm_policy


POLICIES = {
    "random": random_baseline,
    "idm": idm_baseline,
}


def evaluate(build, seeds) -> dict:
    env, policy = build()
    try:
        rows = [run_episode(env, policy, seed) for seed in seeds]
    finally:
        env.close()
    ttcs = [row["min_ttc"] for row in rows]
    return {
        "episodes": len(rows),
        "mean_return": statistics.mean(row["return"] for row in rows),
        "crash_rate": statistics.mean(row["crashed"] for row in rows),
        "median_min_ttc": statistics.median(ttcs),
        "never_closed": sum(math.isinf(t) for t in ttcs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=20)
    # Learners train on seeds 0, 1, 2, ...; evaluate on seeds they never saw,
    # matching scripts/train_model_based.py.
    parser.add_argument("--first-seed", type=int, default=10_000)
    args = parser.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.episodes)

    print(f"{SCENARIO}, {args.episodes} episodes per policy, seeds {seeds.start}-{seeds.stop - 1}")
    print(f"{'policy':<8} {'return':>8} {'crash':>7} {'min TTC (median)':>17}")
    for name, build in POLICIES.items():
        r = evaluate(build, seeds)
        ttc = r["median_min_ttc"]
        ttc_text = "none closing" if math.isinf(ttc) else f"{ttc:.2f} s"
        print(f"{name:<8} {r['mean_return']:>8.2f} {r['crash_rate']:>7.0%} {ttc_text:>17}")


if __name__ == "__main__":
    main()
