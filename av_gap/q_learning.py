"""Model-free tabular Q-learning for av_gap.

Keeps one value per (state bin, action) and updates it from experience only.
It never counts transitions, so no model of what happens next is built.
"""

import numpy as np

from av_gap.discretize import encode, n_states

N_ACTIONS = 5
ALPHA = 0.1   # learning rate: how far each experience moves the estimate
GAMMA = 0.95  # discount: how much the next bin's value counts (matches model_based)
EPS_START = 1.0  # first step: almost all actions are random
EPS_END = 0.05   # end of budget: mostly greedy, a little exploration stays


def epsilon_at(step, total_steps):
    """Return epsilon at `step`: falls linearly from EPS_START to EPS_END over total_steps, then holds."""
    frac = min(step / max(total_steps, 1), 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(Q, s, eps, rng):
    """Epsilon-greedy: random action with prob eps, else the best action in Q[s]."""
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS))
    # Break ties at random so an all-zero row doesn't always pick action 0
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def make_q_table():
    """Return a zeroed Q-table of shape (n_states, N_ACTIONS): one row per bin, one column per action."""
    return np.zeros((n_states, N_ACTIONS))


def update(Q, s, a, r, s_next, done, alpha=ALPHA, gamma=GAMMA):
    """Apply one Q-learning update to Q[s, a] in place.

    Moves Q[s, a] toward r + gamma * max(Q[s_next]). If `done` (a crash), there is
    no next bin to look ahead to, so the target is just r. Pass `terminated`, not
    `truncated`, as `done`: a time-out still has a real next state.
    """
    # Target: reward now, plus the discounted best value from the bin we landed in
    target = r if done else r + gamma * Q[s_next].max()
    # Step a fraction alpha of the way from the old estimate to the target
    Q[s, a] += alpha * (target - Q[s, a])
