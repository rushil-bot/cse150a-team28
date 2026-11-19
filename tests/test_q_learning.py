"""Checks on the Q-table and the single Q-learning update (sub-issue 3.1)."""

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
