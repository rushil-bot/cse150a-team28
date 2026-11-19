"""Train and evaluate the model-based RL agent (VI or PI) on the roundabout.

Examples:
    python scripts/train_model_based.py
    python scripts/train_model_based.py --method pi --steps 3000 --episodes 20
"""

from __future__ import annotations

import argparse

from av_gap.envs import make_env
from av_gap.model_based import train
from av_gap.rollout import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate model-based RL.")
    parser.add_argument("--scenario", default="roundabout", choices=("roundabout", "highway"))
    parser.add_argument("--method", default="vi", choices=("vi", "pi"))
    parser.add_argument("--steps", type=int, default=2000, help="Environment step budget")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--collision-reward", type=float, default=None)
    parser.add_argument("--high-speed-reward", type=float, default=None)
    args = parser.parse_args()

    reward: dict[str, float] = {}
    if args.collision_reward is not None:
        reward["collision_reward"] = args.collision_reward
    if args.high_speed_reward is not None:
        reward["high_speed_reward"] = args.high_speed_reward

    policy = train(
        scenario=args.scenario,
        reward=reward or None,
        step_budget=args.steps,
        seed=args.seed,
        method=args.method,
        gamma=args.gamma,
    )

    env = make_env(args.scenario, reward=reward or None)
    try:
        eval_seeds = range(10_000, 10_000 + args.episodes)
        rows = [run_episode(env, policy, seed) for seed in eval_seeds]
    finally:
        env.close()

    mean_return = sum(row["return"] for row in rows) / len(rows)
    mean_length = sum(row["length"] for row in rows) / len(rows)
    crash_rate = sum(row["crashed"] for row in rows) / len(rows)
    print(
        f"{args.scenario} ({args.method}, {args.steps} train steps): "
        f"mean return {mean_return:.3f}, mean length {mean_length:.1f}, "
        f"crash rate {crash_rate:.0%} over {len(rows)} eval seeds"
    )


if __name__ == "__main__":
    main()
