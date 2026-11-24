"""Figures for the report, drawn from saved drives (see `av_gap.evaluation`).

Everything here reads rows, never runs the simulator, so figures can be redrawn
from `results/*.csv` without retraining (`scripts/plot.py`).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from av_gap.evaluation import AGENT_NAMES, summarize

# Same color per learner in every figure. PI is dashed so it stays visible when
# it lands exactly on VI (they often find the same policy).
STYLE = {
    "vi": {"color": "tab:green", "marker": "o"},
    "pi": {"color": "tab:blue", "marker": "x", "linestyle": "--", "zorder": 3},
    "q": {"color": "tab:orange", "marker": "o"},
}


def _budget_label(steps: int) -> str:
    return f"{steps / 1000:g}k"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # write files only; no window
    import matplotlib.pyplot as plt

    return plt


def plot_learning_curves(rows: list[dict], out_dir: str | Path) -> list[Path]:
    """Return and crash rate against training steps, one line per learner.

    Each point averages over training seeds; the band spans the lowest and
    highest training seed. Baselines are flat dashed lines on the same axes.
    """
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    learners = [r for r in rows if r["budget"] is not None]
    baselines = [r for r in rows if r["budget"] is None]
    per_seed = summarize(learners, ("kind", "budget", "train_seed"))
    base = summarize(baselines, ("kind",))

    paths = []
    for metric, ylabel, fname in (
        ("mean_return", "Evaluation return", "curves_return.png"),
        ("crash_rate", "Crash rate", "curves_crash.png"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        by_kind = defaultdict(lambda: defaultdict(list))
        for s in per_seed:
            by_kind[s["kind"]][s["budget"]].append(s[metric])
        for kind, budgets in sorted(by_kind.items()):
            xs = sorted(budgets)
            mean = [statistics.mean(budgets[x]) for x in xs]
            low = [min(budgets[x]) for x in xs]
            high = [max(budgets[x]) for x in xs]
            style = STYLE.get(kind, {"marker": "o"})
            (line,) = ax.plot(xs, mean, label=AGENT_NAMES[kind], **style)
            ax.fill_between(xs, low, high, color=line.get_color(), alpha=0.15)
        for s in base:
            ax.axhline(s[metric], linestyle="--", color="gray" if s["kind"] == "random" else "black",
                       label=AGENT_NAMES[s["kind"]])
        ax.set_xscale("log")
        budgets = sorted({s["budget"] for s in per_seed})
        ax.set_xticks(budgets, [_budget_label(b) for b in budgets])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Training steps (env.step calls)")
        ax.set_ylabel(ylabel)
        if metric == "crash_rate":
            ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Roundabout, equal step budgets")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_sweep(rows: list[dict], out_dir: str | Path) -> Path:
    """Crash rate against progress (distance), one point per collision weight."""
    plt = _pyplot()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, ("kind", "collision_reward", "high_speed_reward"))
    # Spread across training seeds, drawn as range bars when there is more than one.
    spread = defaultdict(list)
    for s in summarize(rows, ("kind", "collision_reward", "high_speed_reward", "train_seed")):
        spread[(s["kind"], s["collision_reward"], s["high_speed_reward"])].append(s)

    fig, ax = plt.subplots(figsize=(6, 4))
    by_kind = defaultdict(list)
    for s in summary:
        by_kind[s["kind"]].append(s)
    for kind, points in sorted(by_kind.items()):
        points.sort(key=lambda s: s["collision_reward"])
        xs = [s["mean_distance"] for s in points]
        ys = [s["crash_rate"] for s in points]
        (line,) = ax.plot(xs, ys, label=AGENT_NAMES[kind], **STYLE.get(kind, {"marker": "o"}))
        for s, x, y in zip(points, xs, ys):
            runs = spread[(s["kind"], s["collision_reward"], s["high_speed_reward"])]
            if len(runs) > 1:
                dist = [r["mean_distance"] for r in runs]
                crash = [r["crash_rate"] for r in runs]
                ax.errorbar(x, y, xerr=[[x - min(dist)], [max(dist) - x]],
                            yerr=[[y - min(crash)], [max(crash) - y]],
                            color=line.get_color(), alpha=0.4, capsize=2, linewidth=1)
        for i, (s, x, y) in enumerate(zip(points, xs, ys)):
            # Alternate labels above and below so neighboring weights don't collide.
            ax.annotate(f"{s['collision_reward']:g}", (x, y), textcoords="offset points",
                        xytext=(0, 7 if i % 2 == 0 else -12), ha="center", fontsize=7,
                        color=line.get_color())
    ax.set_xlabel("Progress: mean distance per drive (m)")
    ax.set_ylabel("Crash rate")
    top = max(r["crash_rate"] for runs in spread.values() for r in runs)
    ax.set_ylim(-0.05, min(1.0, top + 0.15))
    ax.legend(fontsize=8)
    seeds = {r["train_seed"] for r in rows}
    note = f"; bars: range over {len(seeds)} training seeds" if len(seeds) > 1 else ""
    ax.set_title(f"Collision weight vs. progress\n(labels: collision_reward{note})", fontsize=10)
    fig.tight_layout()
    path = out_dir / "sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
