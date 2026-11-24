"""Experiment harness for evaluations: train policies, score them, save and summarize.

Every experiment (the final table, the reward sweep, the learning curves) is
built from the same pieces:

- A `PolicySpec` names one policy: a baseline, or a learner with its training
  scenario, step budget, training seed, and reward weights.
- `evaluate_spec` trains it (if it learns) and runs it through `run_episode`
  on every requested evaluation scenario, returning one row per drive.
- `run_specs` does that for many specs in parallel processes.
- `write_rows` / `read_rows` save drives to CSV; `summarize` and `format_table`
  turn them into the numbers that go in the report.

Seeds. Evaluation drives use seeds from `EVAL_FIRST_SEED` (10,000) up. Both learners reset training episodes with
`seed + episode_number`, so a run with training seed `s` and budget `b` resets
on seeds inside `[s, s + b)`. Training seeds start at 20,000, which keeps that
range clear of the evaluation seeds for any budget. `check_seeds` enforces it.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from av_gap import model_based, q_learning
from av_gap.envs import make_env
from av_gap.rollout import IDMDriven, idm_policy, random_policy, run_episode

# Where a policy can be evaluated. Keys are what the command line accepts.
EVAL_SCENARIOS = {
    "roundabout": {"scenario": "roundabout"},
    "roundabout-calibrated": {"scenario": "roundabout", "calibrated_traffic": True},
    "highway": {"scenario": "highway"},
}

# Where a learner can be trained. Calibrated traffic is for evaluation only:
# neither `train(...)` function takes a `calibrated_traffic` option.
TRAIN_SCENARIOS = ("roundabout", "highway")

BASELINES = ("random", "idm")
LEARNERS = ("vi", "pi", "q")
AGENT_NAMES = {
    "random": "Random baseline",
    "idm": "Built-in IDM",
    "vi": "Model-based (VI)",
    "pi": "Model-based (PI)",
    "q": "Q-learning",
}

EVAL_FIRST_SEED = 10_000
TRAIN_SEEDS = (20_000, 120_000, 220_000)

METRICS = ("return", "length", "crashed", "min_ttc", "distance", "avg_speed", "decision_time")


@dataclass(frozen=True)
class PolicySpec:
    """One policy to evaluate. Baselines leave the training fields empty."""

    kind: str
    train_scenario: str | None = None
    budget: int | None = None
    train_seed: int | None = None
    collision_reward: float | None = None
    high_speed_reward: float | None = None

    def __post_init__(self):
        if self.kind not in BASELINES + LEARNERS:
            raise ValueError(f"Unknown policy kind {self.kind!r}")
        if self.kind in LEARNERS:
            if self.train_scenario not in TRAIN_SCENARIOS:
                raise ValueError(
                    f"Learners train on {TRAIN_SCENARIOS}, got {self.train_scenario!r}"
                )
            if self.budget is None or self.train_seed is None:
                raise ValueError("Learners need a budget and a training seed")

    @property
    def is_learner(self) -> bool:
        return self.kind in LEARNERS

    @property
    def reward(self) -> dict | None:
        """Reward overrides for `make_env`, or None for the scenario defaults."""
        reward = {}
        if self.collision_reward is not None:
            reward["collision_reward"] = self.collision_reward
        if self.high_speed_reward is not None:
            reward["high_speed_reward"] = self.high_speed_reward
        return reward or None

    @property
    def label(self) -> str:
        name = AGENT_NAMES[self.kind]
        if not self.is_learner:
            return name
        return f"{name}, {self.budget / 1000:g}k on {self.train_scenario}"


def check_seeds(spec: PolicySpec, eval_seeds: Sequence[int]) -> None:
    """Fail if training could reset on an evaluation seed."""
    if not spec.is_learner or not eval_seeds:
        return
    train_lo, train_hi = spec.train_seed, spec.train_seed + spec.budget
    eval_lo, eval_hi = min(eval_seeds), max(eval_seeds) + 1
    if train_lo < eval_hi and eval_lo < train_hi:
        raise ValueError(
            f"{spec.label} with training seed {spec.train_seed} may reset on seeds "
            f"{train_lo}-{train_hi - 1}, which overlaps evaluation seeds "
            f"{eval_lo}-{eval_hi - 1}. Pick a training seed outside that range."
        )


def train_policy(spec: PolicySpec):
    """Train a learner and return its greedy policy."""
    if spec.kind in ("vi", "pi"):
        return model_based.train(
            spec.train_scenario,
            spec.reward,
            spec.budget,
            spec.train_seed,
            method=spec.kind,
        )
    if spec.kind == "q":
        return q_learning.train(spec.train_scenario, spec.reward, spec.budget, spec.train_seed)
    raise ValueError(f"{spec.kind!r} does not train")


def evaluate_spec(
    spec: PolicySpec, eval_scenarios: Sequence[str], seeds: Sequence[int]
) -> list[dict]:
    """Train `spec` once (if it learns), then drive it on every evaluation scenario.

    Evaluation uses the same reward weights the policy trained with, so
    `return` is comparable to its training objective. Crash rate, distance,
    and time-to-collision do not depend on the weights.
    """
    check_seeds(spec, seeds)
    started = time.perf_counter()
    learned = train_policy(spec) if spec.is_learner else None
    train_seconds = time.perf_counter() - started

    rows = []
    for name in eval_scenarios:
        env = make_env(**EVAL_SCENARIOS[name], reward=spec.reward)
        if spec.kind == "idm":
            env, policy = IDMDriven(env), idm_policy
        else:
            policy = learned
        try:
            for seed in seeds:
                if spec.kind == "random":
                    # Seed the random actions too, so the baseline is repeatable.
                    policy = random_policy(env, seed=seed)
                result = run_episode(env, policy, seed)
                rows.append(
                    {
                        "agent": spec.label,
                        "kind": spec.kind,
                        "train_scenario": spec.train_scenario,
                        "budget": spec.budget,
                        "train_seed": spec.train_seed,
                        "collision_reward": spec.collision_reward,
                        "high_speed_reward": spec.high_speed_reward,
                        "eval_scenario": name,
                        "seed": seed,
                        "train_seconds": train_seconds,
                        **result,
                    }
                )
        finally:
            env.close()
    return rows


def _job_cost(spec: PolicySpec, eval_scenarios: Sequence[str]) -> float:
    """Rough relative cost, so the longest jobs start first. Highway is ~10x slower."""
    def per_step(scenario):
        return 10.0 if scenario == "highway" else 1.0

    cost = (spec.budget or 0) * per_step(spec.train_scenario)
    cost += sum(per_step(EVAL_SCENARIOS[n]["scenario"]) * 4_000 for n in eval_scenarios)
    return cost


def default_workers() -> int:
    """Leave a couple of cores free for the rest of the machine."""
    return max(1, (os.cpu_count() or 2) - 2)


def run_specs(
    specs: Sequence[PolicySpec],
    eval_scenarios: Sequence[str],
    seeds: Sequence[int],
    workers: int | None = None,
    log=print,
) -> list[dict]:
    """Evaluate every spec, in parallel processes, and return all drives."""
    for spec in specs:
        check_seeds(spec, seeds)
    for name in eval_scenarios:
        if name not in EVAL_SCENARIOS:
            raise ValueError(f"Unknown evaluation scenario {name!r}")
    ordered = sorted(specs, key=lambda s: -_job_cost(s, eval_scenarios))
    workers = workers or default_workers()
    started = time.perf_counter()
    rows: list[dict] = []
    if workers == 1:
        for i, spec in enumerate(ordered, 1):
            rows += evaluate_spec(spec, eval_scenarios, seeds)
            log(f"  [{i}/{len(ordered)}] {spec.label} ({time.perf_counter() - started:.0f} s)")
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_spec, spec, eval_scenarios, seeds): spec
            for spec in ordered
        }
        for i, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            log(
                f"  [{i}/{len(ordered)}] {futures[future].label} "
                f"({time.perf_counter() - started:.0f} s)"
            )
    return rows


# ----------------------------------------------------------------------------
# Saving and loading

FIELDS = (
    "agent",
    "kind",
    "train_scenario",
    "budget",
    "train_seed",
    "collision_reward",
    "high_speed_reward",
    "eval_scenario",
    "seed",
    "train_seconds",
) + METRICS

_INT_FIELDS = {"budget", "train_seed", "seed", "length"}
_FLOAT_FIELDS = {
    "collision_reward",
    "high_speed_reward",
    "train_seconds",
    "return",
    "min_ttc",
    "distance",
    "avg_speed",
    "decision_time",
}


def write_rows(rows: Iterable[dict], path: str | Path) -> Path:
    """Save one row per drive as CSV. Missing values are written as blanks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row[k] for k in FIELDS})
    return path


def read_rows(path: str | Path) -> list[dict]:
    """Load drives saved by `write_rows`, restoring numbers and booleans."""
    rows = []
    with Path(path).open(newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in _INT_FIELDS:
                    row[key] = int(value)
                elif key in _FLOAT_FIELDS:
                    row[key] = float(value)  # float("inf") reads back infinity
                elif key == "crashed":
                    row[key] = value == "True"
                else:
                    row[key] = value
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Summaries


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Behaves well near 0% and 100%."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(rows: Iterable[dict], by: Sequence[str]) -> list[dict]:
    """Aggregate drives into one row per group of the `by` columns.

    Crash rate comes with a Wilson interval, min TTC is the median over
    drives (it can be infinite), and the rest are means.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in by)].append(row)
    summary = []
    for key, group in groups.items():
        crashes = sum(r["crashed"] for r in group)
        low, high = wilson_interval(crashes, len(group))
        summary.append(
            {
                **dict(zip(by, key)),
                "drives": len(group),
                "crash_rate": crashes / len(group),
                "crash_low": low,
                "crash_high": high,
                "median_min_ttc": statistics.median(r["min_ttc"] for r in group),
                "mean_decision_time": statistics.mean(r["decision_time"] for r in group),
                "mean_return": statistics.mean(r["return"] for r in group),
                "mean_distance": statistics.mean(r["distance"] for r in group),
                "mean_speed": statistics.mean(r["avg_speed"] for r in group),
            }
        )
    return summary


KIND_ORDER = BASELINES + LEARNERS  # random, IDM, then the learners


def display_order(summary_row: dict) -> tuple:
    """Sort key putting baselines first (random, then IDM), then learners by budget.

    The summary must be grouped by `kind` (and `budget`, if present).
    """
    return (KIND_ORDER.index(summary_row["kind"]), summary_row.get("budget") or 0)


def _ttc_text(seconds: float) -> str:
    return "none closing" if math.isinf(seconds) else f"{seconds:.2f}"


def format_table(summary: Sequence[dict], label_keys: Sequence[str]) -> str:
    """A plain-text table in the same columns as the report's results table."""
    header = (
        [*label_keys, "drives", "crash % [95% CI]", "min TTC (s)", "decision (s)",
         "return", "dist. (m)", "speed (m/s)"]
    )
    body = []
    for s in summary:
        body.append(
            [
                *(str(s[k]) for k in label_keys),
                str(s["drives"]),
                f"{100 * s['crash_rate']:.1f} [{100 * s['crash_low']:.1f}, "
                f"{100 * s['crash_high']:.1f}]",
                _ttc_text(s["median_min_ttc"]),
                f"{s['mean_decision_time']:.6f}",
                f"{s['mean_return']:.2f}",
                f"{s['mean_distance']:.1f}",
                f"{s['mean_speed']:.2f}",
            ]
        )
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in body]
    return "\n".join(lines)
