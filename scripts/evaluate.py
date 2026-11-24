"""Run the Ta experiments. Each one saves its drives to results/ as CSV.

    python scripts/evaluate.py                 # quick baseline check (20 drives each)
    python scripts/evaluate.py table           # 5.5: the 500-drive results table
    python scripts/evaluate.py sweep           # 5.4: collision weight vs. progress
    python scripts/evaluate.py curves          # 5.3: learning curves at equal budgets

`scripts/plot.py` redraws every figure and table from the saved CSVs without rerunning anything.

"""

from __future__ import annotations

import argparse
from pathlib import Path

from av_gap.evaluation import (
    BASELINES,
    EVAL_FIRST_SEED,
    EVAL_SCENARIOS,
    LEARNERS,
    TRAIN_SCENARIOS,
    TRAIN_SEEDS,
    PolicySpec,
    default_workers,
    display_order,
    format_table,
    run_specs,
    summarize,
    write_rows,
)

RESULTS = Path("results")


def _seeds(args) -> range:
    return range(args.first_seed, args.first_seed + args.episodes)


def _add_common(parser, episodes: int, out: str) -> None:
    parser.add_argument("--episodes", type=int, default=episodes,
                        help=f"evaluation drives per policy per scenario (default {episodes})")
    parser.add_argument("--first-seed", type=int, default=EVAL_FIRST_SEED,
                        help=f"first evaluation seed (default {EVAL_FIRST_SEED})")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="parallel processes (default: CPU cores minus 2)")
    parser.add_argument("--out", type=Path, default=RESULTS / out,
                        help=f"CSV to write (default results/{out})")


def _check_train_scenarios(names) -> None:
    for name in names:
        if name == "roundabout-calibrated":
            raise SystemExit(
                "Training on calibrated traffic needs a calibrated_traffic option on "
                "both train(...) functions (a shared-contract change). It is not "
                "available yet; evaluate on roundabout-calibrated instead."
            )


def cmd_baselines(args) -> None:
    seeds = _seeds(args)
    specs = [PolicySpec(kind) for kind in BASELINES]
    rows = run_specs(specs, ["roundabout"], seeds, workers=args.workers)
    write_rows(rows, args.out)
    summary = sorted(summarize(rows, ("kind", "agent")), key=display_order)
    print(format_table(summary, ("agent",)))


def cmd_table(args) -> None:
    _check_train_scenarios(args.train)
    seeds = _seeds(args)
    specs = [PolicySpec(kind) for kind in BASELINES]
    specs += [
        PolicySpec(method, train_scenario=scenario, budget=budget, train_seed=args.train_seed)
        for scenario in args.train
        for method in args.methods
        for budget in args.budgets
    ]
    print(f"{len(specs)} policies x {len(args.eval)} scenarios x {len(seeds)} drives "
          f"(seeds {seeds.start}-{seeds.stop - 1}), {args.workers} workers")
    rows = run_specs(specs, args.eval, seeds, workers=args.workers)
    print(f"saved {write_rows(rows, args.out)}\n")
    keys = ("eval_scenario", "agent")
    summary = sorted(summarize(rows, ("eval_scenario", "kind", "budget", "agent")),
                     key=lambda s: (s["eval_scenario"], display_order(s), s["agent"]))
    print(format_table(summary, keys))


def cmd_sweep(args) -> None:
    seeds = _seeds(args)
    specs = [
        PolicySpec(method, train_scenario="roundabout", budget=args.budget,
                   train_seed=seed, collision_reward=weight,
                   high_speed_reward=args.high_speed)
        for method in args.methods
        for weight in args.collision
        for seed in args.train_seeds
    ]
    print(f"{len(specs)} trained policies x {len(seeds)} drives, {args.workers} workers")
    rows = run_specs(specs, ["roundabout"], seeds, workers=args.workers)
    print(f"saved {write_rows(rows, args.out)}\n")
    keys = ("kind", "collision_reward")
    summary = sorted(summarize(rows, keys), key=lambda s: (display_order(s), -s["collision_reward"]))
    print(format_table(summary, keys))
    from av_gap.plots import plot_sweep

    print(f"\nplot: {plot_sweep(rows, args.out.parent)}")


def cmd_curves(args) -> None:
    seeds = _seeds(args)
    specs = [PolicySpec(kind) for kind in BASELINES]
    specs += [
        PolicySpec(method, train_scenario="roundabout", budget=budget, train_seed=seed)
        for method in args.methods
        for budget in args.budgets
        for seed in args.train_seeds
    ]
    print(f"{len(specs)} policies x {len(seeds)} drives, {args.workers} workers")
    rows = run_specs(specs, ["roundabout"], seeds, workers=args.workers)
    print(f"saved {write_rows(rows, args.out)}\n")
    keys = ("kind", "budget")
    summary = sorted(summarize(rows, keys), key=display_order)
    print(format_table(summary, keys))
    from av_gap.plots import plot_learning_curves

    for path in plot_learning_curves(rows, args.out.parent):
        print(f"plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("baselines", help="random vs. IDM on the default roundabout")
    _add_common(p, episodes=20, out="baselines.csv")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("table", help="5.5: final results table")
    _add_common(p, episodes=500, out="table.csv")
    p.add_argument("--train", nargs="+", default=["roundabout"],
                   choices=[*TRAIN_SCENARIOS, "roundabout-calibrated"],
                   help="scenarios to train learners on (default: roundabout)")
    p.add_argument("--eval", nargs="+", default=["roundabout", "roundabout-calibrated"],
                   choices=list(EVAL_SCENARIOS),
                   help="scenarios to evaluate on (default: both roundabouts)")
    p.add_argument("--budgets", nargs="+", type=int, default=[5_000, 20_000])
    p.add_argument("--methods", nargs="+", default=list(LEARNERS), choices=LEARNERS)
    p.add_argument("--train-seed", type=int, default=TRAIN_SEEDS[0])
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("sweep", help="5.4: collision weight vs. progress")
    _add_common(p, episodes=100, out="sweep.csv")
    p.add_argument("--collision", nargs="+", type=float, default=[-0.5, -1, -2, -5, -10],
                   help="collision_reward values to try (roundabout default: -1)")
    p.add_argument("--high-speed", type=float, default=None,
                   help="high_speed_reward, held fixed (default: the scenario's own, 0.2)")
    p.add_argument("--budget", type=int, default=20_000)
    p.add_argument("--methods", nargs="+", default=["vi", "q"], choices=LEARNERS)
    p.add_argument("--train-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS),
                   help="training runs per weight, averaged in the plot")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("curves", help="5.3: learning curves at equal step budgets")
    _add_common(p, episodes=20, out="curves.csv")
    p.add_argument("--budgets", nargs="+", type=int,
                   default=[2_000, 5_000, 10_000, 20_000, 50_000])
    p.add_argument("--train-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    p.add_argument("--methods", nargs="+", default=list(LEARNERS), choices=LEARNERS)
    p.set_defaults(func=cmd_curves)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["baselines"])
    args.func(args)


if __name__ == "__main__":
    main()
