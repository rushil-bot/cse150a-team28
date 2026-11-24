"""Checks on the experiment harness: seeds, saving, summaries, and a tiny run."""

import math

import pytest

from av_gap.evaluation import (
    EVAL_FIRST_SEED,
    TRAIN_SEEDS,
    PolicySpec,
    check_seeds,
    evaluate_spec,
    read_rows,
    run_specs,
    summarize,
    wilson_interval,
    write_rows,
)
from av_gap.rollout import run_episode


def test_default_training_seeds_never_touch_evaluation_seeds():
    """Training resets on [seed, seed + budget); evaluation must stay outside it."""
    eval_seeds = range(EVAL_FIRST_SEED, EVAL_FIRST_SEED + 500)
    for seed in TRAIN_SEEDS:
        check_seeds(PolicySpec("q", "roundabout", budget=100_000, train_seed=seed), eval_seeds)


def test_overlapping_seeds_are_rejected():
    spec = PolicySpec("vi", "roundabout", budget=50_000, train_seed=0)
    with pytest.raises(ValueError, match="overlaps"):
        check_seeds(spec, range(EVAL_FIRST_SEED, EVAL_FIRST_SEED + 20))


def test_learners_need_a_known_training_scenario():
    with pytest.raises(ValueError):
        PolicySpec("q", "roundabout-calibrated", budget=100, train_seed=TRAIN_SEEDS[0])
    with pytest.raises(ValueError):
        PolicySpec("q", "roundabout")  # no budget or seed


def test_wilson_interval_contains_the_rate_and_handles_zero():
    low, high = wilson_interval(12, 100)
    assert low < 0.12 < high
    low, high = wilson_interval(0, 500)
    assert low == 0.0 and 0 < high < 0.01


def test_rows_survive_a_csv_round_trip(tmp_path):
    rows = [
        {"agent": "Random baseline", "kind": "random", "eval_scenario": "roundabout",
         "seed": 10_000, "return": 7.5, "length": 11, "crashed": True,
         "min_ttc": math.inf, "distance": 60.2, "avg_speed": 7.1, "decision_time": 1e-5,
         "train_seconds": 0.0},
    ]
    back = read_rows(write_rows(rows, tmp_path / "rows.csv"))
    assert back[0]["crashed"] is True
    assert math.isinf(back[0]["min_ttc"])
    assert back[0]["seed"] == 10_000
    assert back[0]["budget"] is None


def test_summary_counts_crashes_and_takes_the_median_ttc():
    rows = [
        {"agent": "a", "crashed": c, "min_ttc": t, "decision_time": 1e-5, "return": 1.0,
         "distance": 10.0, "avg_speed": 5.0}
        for c, t in [(True, 0.5), (False, 2.0), (False, math.inf)]
    ]
    (s,) = summarize(rows, ("agent",))
    assert s["drives"] == 3
    assert s["crash_rate"] == pytest.approx(1 / 3)
    assert s["median_min_ttc"] == 2.0


def test_random_baseline_is_repeatable():
    spec = PolicySpec("random")
    first = evaluate_spec(spec, ["roundabout"], [EVAL_FIRST_SEED])
    second = evaluate_spec(spec, ["roundabout"], [EVAL_FIRST_SEED])
    assert first[0]["return"] == second[0]["return"]
    assert first[0]["distance"] == second[0]["distance"]


def test_tiny_run_covers_baselines_and_learners():
    specs = [
        PolicySpec("idm"),
        PolicySpec("q", "roundabout", budget=30, train_seed=TRAIN_SEEDS[0]),
    ]
    seeds = [EVAL_FIRST_SEED, EVAL_FIRST_SEED + 1]
    rows = run_specs(specs, ["roundabout", "roundabout-calibrated"], seeds, workers=1,
                     log=lambda _msg: None)
    assert len(rows) == len(specs) * 2 * len(seeds)
    assert {r["eval_scenario"] for r in rows} == {"roundabout", "roundabout-calibrated"}
    assert all(r["decision_time"] > 0 for r in rows)


def test_harness_scores_drives_exactly_like_run_episode():
    """The harness must not change the numbers, only collect them."""
    from av_gap.envs import make_env
    from av_gap.rollout import IDMDriven, idm_policy

    (row,) = evaluate_spec(PolicySpec("idm"), ["roundabout"], [EVAL_FIRST_SEED])
    env = IDMDriven(make_env())
    try:
        direct = run_episode(env, idm_policy, EVAL_FIRST_SEED)
    finally:
        env.close()
    for key in ("return", "length", "crashed", "min_ttc", "distance", "avg_speed"):
        assert row[key] == direct[key]
