"""Redraw every figure and table from saved drives, without rerunning anything.

    python scripts/plot.py               # reads results/
    python scripts/plot.py --dir other/  # or another folder

Looks for the CSVs `scripts/evaluate.py` writes (table.csv, sweep.csv,
curves.csv) and redraws whichever exist. Figures go next to the CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from av_gap.evaluation import display_order, format_table, read_rows, summarize
from av_gap.plots import plot_learning_curves, plot_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    found = False
    table = args.dir / "table.csv"
    if table.exists():
        found = True
        keys = ("eval_scenario", "agent")
        summary = sorted(summarize(read_rows(table), ("eval_scenario", "kind", "budget", "agent")),
                         key=lambda s: (s["eval_scenario"], display_order(s), s["agent"]))
        print(f"== {table}\n{format_table(summary, keys)}\n")

    sweep = args.dir / "sweep.csv"
    if sweep.exists():
        found = True
        rows = read_rows(sweep)
        keys = ("kind", "collision_reward")
        summary = sorted(summarize(rows, keys), key=lambda s: (display_order(s), -s["collision_reward"]))
        print(f"== {sweep}\n{format_table(summary, keys)}")
        print(f"plot: {plot_sweep(rows, args.dir)}\n")

    curves = args.dir / "curves.csv"
    if curves.exists():
        found = True
        rows = read_rows(curves)
        keys = ("kind", "budget")
        summary = sorted(summarize(rows, keys), key=display_order)
        print(f"== {curves}\n{format_table(summary, keys)}")
        for path in plot_learning_curves(rows, args.dir):
            print(f"plot: {path}")

    if not found:
        raise SystemExit(f"No results CSVs in {args.dir}/. Run scripts/evaluate.py first.")


if __name__ == "__main__":
    main()
