"""Checks on the Q-table, the single Q-learning update (sub-issue 3.1), and a budget sanity check."""

import pytest

from av_gap.discretize import n_states
from av_gap.q_learning import make_q_table, update


def test_q_table_shape_and_zero_start():
    """The table has one row per bin, one column per action, and starts at zero."""
    Q = make_q_table()
    assert Q.shape == (n_states, 5)
    assert (Q == 0).all()


def test_update_moves_by_alpha_times_gap():
    """One non-terminal update moves Q[s, a] by exactly alpha * (target - old)."""
    Q = make_q_table()
    s, a, s_next = 3, 2, 7
    Q[s, a] = 1.0
    Q[s_next] = [0.5, 4.0, -1.0, 2.0, 0.0]  # best next value is 4.0
    r, alpha, gamma = 1.5, 0.1, 0.9

    target = r + gamma * 4.0
    expected = 1.0 + alpha * (target - 1.0)
    update(Q, s, a, r, s_next, done=False, alpha=alpha, gamma=gamma)

    assert Q[s, a] == pytest.approx(expected)


def test_done_drops_lookahead():
    """On a crash the next bin's values are ignored and the target is just r."""
    Q = make_q_table()
    s, a, s_next = 3, 2, 7
    Q[s, a] = 1.0
    Q[s_next] = 100.0  # would dominate the target if it were used
    r, alpha = -1.0, 0.1

    update(Q, s, a, r, s_next, done=True, alpha=alpha)

    assert Q[s, a] == pytest.approx(1.0 + alpha * (r - 1.0))


def test_epsilon_endpoints_and_hold():
    """Epsilon starts at EPS_START, ends at EPS_END, and stays there past the budget."""
    from av_gap.q_learning import EPS_END, EPS_START, epsilon_at
    assert epsilon_at(0, 100) == EPS_START
    assert epsilon_at(100, 100) == pytest.approx(EPS_END)
    assert epsilon_at(500, 100) == pytest.approx(EPS_END)


def test_epsilon_falls():
    """Epsilon never rises and ends far below where it started."""
    from av_gap.q_learning import epsilon_at
    vals = [epsilon_at(t, 100) for t in range(0, 101, 10)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] - vals[-1] > 0.9


def test_select_action_greedy_and_random():
    """eps=0 picks the argmax; eps=1 reaches every action."""
    import numpy as np
    from av_gap.q_learning import select_action
    Q = make_q_table()
    Q[0] = [0, 0, 3.0, 0, 0]
    rng = np.random.default_rng(0)
    assert select_action(Q, 0, 0.0, rng) == 2
    assert {select_action(Q, 0, 1.0, rng) for _ in range(300)} == {0, 1, 2, 3, 4}


def test_select_action_same_seed_same_actions():
    """Two RNGs with the same seed give identical action sequences."""
    import numpy as np
    from av_gap.q_learning import select_action
    Q = make_q_table()
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(7)
        runs.append([select_action(Q, 0, 0.5, rng) for _ in range(50)])
    assert runs[0] == runs[1]


def _counting_make_env(counter):
    """Return a make_env replacement whose env counts step calls in counter[0]."""
    from av_gap.envs import make_env as real

    def factory(*args, **kwargs):
        env = real(*args, **kwargs)
        orig = env.step

        def step(a):
            counter[0] += 1
            return orig(a)

        env.step = step
        return env

    return factory


@pytest.mark.parametrize("budget", [30, 47])
def test_train_step_count_equals_budget(monkeypatch, budget):
    """env.step is called exactly step_budget times, even if cut mid-episode."""
    import av_gap.q_learning as ql
    counter = [0]
    monkeypatch.setattr(ql, "make_env", _counting_make_env(counter))
    ql.train(step_budget=budget)
    assert counter[0] == budget


def test_train_policy_valid_and_deterministic():
    """Policy returns an int in 0..4 and gives the same action every call."""
    from av_gap.envs import make_env
    from av_gap.q_learning import train
    policy = train(step_budget=30)
    obs, _ = make_env("roundabout").reset(seed=0)
    acts = {policy(obs) for _ in range(20)}
    assert len(acts) == 1 and acts.pop() in range(5)


def test_more_budget_does_not_hurt():
    """Average return with a large budget is at least as high as with a small one.

    Averages over several training seeds and eval seeds, because a single
    training run is noisy at these small budgets.
    """
    import numpy as np
    from av_gap.envs import make_env
    from av_gap.q_learning import train
    from av_gap.rollout import run_episode

    small, large = 50, 1000           # far apart so the gap beats the noise
    train_seeds = [0, 1, 2]
    eval_seeds = range(1000, 1005)    # held out from the training resets (seed + episode)
    env = make_env("roundabout")

    def avg_return(budget):
        """Mean run_episode return over all training seeds and eval seeds."""
        rets = []
        for ts in train_seeds:
            policy = train(step_budget=budget, seed=ts)
            rets += [run_episode(env, policy, s)["return"] for s in eval_seeds]
        return np.mean(rets)

    assert avg_return(large) >= avg_return(small)
