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
