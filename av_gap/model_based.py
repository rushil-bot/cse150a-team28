"""Model-based tabular RL (Value Iteration and Policy Iteration) for av_gap.

Estimates a discrete MDP `(P, R)` from environment rollouts over the binned
state space in `av_gap.discretize`, then solves it with Value Iteration or
Policy Iteration. Every `env.step` call spent collecting transition counts
counts toward `step_budget`; planning passes do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from av_gap.discretize import encode, n_states
from av_gap.envs import make_env
from av_gap.rollout import Policy

IDLE_ACTION = 1
DEFAULT_GAMMA = 0.95


@dataclass
class TabularModel:
    """Empirical MDP estimated from discrete-state environment rollouts.

    `P[s, a, s']` is the empirical probability of transitioning to non-terminal
    state `s'` after taking action `a` in state `s`. Transitions that terminate
    (crashes) are counted in `visit_counts[s, a]` and `terminal_counts[s, a]`
    but do not add mass to `P[s, a, :]`, so `1 - P[s, a].sum()` is the
    empirical crash probability (absorbing terminal state with value 0).
    """

    n_states: int
    n_actions: int
    visit_counts: np.ndarray
    transition_counts: np.ndarray
    terminal_counts: np.ndarray
    reward_sum: np.ndarray
    P: np.ndarray
    R: np.ndarray
    steps_collected: int
    episodes_collected: int


def build_model_from_counts(
    visit_counts: np.ndarray,
    transition_counts: np.ndarray,
    terminal_counts: np.ndarray,
    reward_sum: np.ndarray,
    steps_collected: int = 0,
    episodes_collected: int = 0,
) -> TabularModel:
    """Convert raw `(s, a)` and `(s, a, s')` counts into `P(s'|s, a)` and `R(s, a)`."""
    num_states, num_actions = visit_counts.shape
    P = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    R = np.zeros((num_states, num_actions), dtype=np.float64)

    visited = visit_counts > 0
    P[visited] = transition_counts[visited] / visit_counts[visited, None]
    R[visited] = reward_sum[visited] / visit_counts[visited]

    return TabularModel(
        n_states=num_states,
        n_actions=num_actions,
        visit_counts=visit_counts,
        transition_counts=transition_counts,
        terminal_counts=terminal_counts,
        reward_sum=reward_sum,
        P=P,
        R=R,
        steps_collected=steps_collected,
        episodes_collected=episodes_collected,
    )


def estimate_model(
    env,
    step_budget: int,
    seed: int = 0,
    *,
    num_states: int = n_states,
) -> TabularModel:
    """Collect exactly `step_budget` environment steps and estimate `(P, R)`.

    Uses uniform random discrete actions to populate `(s, a, s')` counts and
    mean rewards. Episodes are seeded deterministically from `seed`.
    """
    if step_budget < 0:
        raise ValueError(f"step_budget must be non-negative, got {step_budget}")

    n_actions = int(env.action_space.n)
    visit_counts = np.zeros((num_states, n_actions), dtype=np.int64)
    transition_counts = np.zeros((num_states, n_actions, num_states), dtype=np.int64)
    terminal_counts = np.zeros((num_states, n_actions), dtype=np.int64)
    reward_sum = np.zeros((num_states, n_actions), dtype=np.float64)

    rng = np.random.default_rng(seed)
    steps = 0
    episodes = 0

    while steps < step_budget:
        obs, _info = env.reset(seed=seed + episodes)
        state = encode(obs)
        episodes += 1
        done = False

        while not done and steps < step_budget:
            action = int(rng.integers(n_actions))
            next_obs, reward, terminated, truncated, _info = env.step(action)
            steps += 1

            visit_counts[state, action] += 1
            reward_sum[state, action] += float(reward)

            if terminated:
                terminal_counts[state, action] += 1
            else:
                next_state = encode(next_obs)
                transition_counts[state, action, next_state] += 1
                state = next_state

            done = bool(terminated or truncated)

    return build_model_from_counts(
        visit_counts=visit_counts,
        transition_counts=transition_counts,
        terminal_counts=terminal_counts,
        reward_sum=reward_sum,
        steps_collected=steps,
        episodes_collected=episodes,
    )


def _greedy_actions(
    Q: np.ndarray,
    visit_counts: np.ndarray | None = None,
    fallback_action: int = IDLE_ACTION,
) -> np.ndarray:
    """Extract greedy actions from `Q`, using `fallback_action` for unvisited states."""
    actions = np.argmax(Q, axis=1).astype(np.int64)
    if visit_counts is not None:
        unvisited_states = visit_counts.sum(axis=1) == 0
        actions[unvisited_states] = fallback_action
    return actions


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    visit_counts: np.ndarray | None = None,
    fallback_action: int = IDLE_ACTION,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve a tabular MDP with Value Iteration.

    Returns `(V, policy_actions, iterations)`.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")

    num_states = P.shape[0]
    V = np.zeros(num_states, dtype=np.float64)

    iterations = 0
    for iterations in range(1, max_iter + 1):
        Q = R + gamma * np.einsum("ijk,k->ij", P, V)
        V_next = np.max(Q, axis=1)
        delta = float(np.max(np.abs(V_next - V)))
        V = V_next
        if delta < tol:
            break

    Q = R + gamma * np.einsum("ijk,k->ij", P, V)
    policy_actions = _greedy_actions(
        Q, visit_counts=visit_counts, fallback_action=fallback_action
    )
    return V, policy_actions, iterations


def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
    max_iter: int = 1_000,
    visit_counts: np.ndarray | None = None,
    fallback_action: int = IDLE_ACTION,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve a tabular MDP with Policy Iteration (exact linear policy evaluation).

    Returns `(V, policy_actions, iterations)`.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")

    num_states = P.shape[0]
    policy_actions = np.full(num_states, fallback_action, dtype=np.int64)
    eye = np.eye(num_states, dtype=np.float64)
    state_idx = np.arange(num_states)
    V = np.zeros(num_states, dtype=np.float64)

    iterations = 0
    for iterations in range(1, max_iter + 1):
        P_pi = P[state_idx, policy_actions]
        R_pi = R[state_idx, policy_actions]
        V = np.linalg.solve(eye - gamma * P_pi, R_pi)

        Q = R + gamma * np.einsum("ijk,k->ij", P, V)
        new_actions = _greedy_actions(
            Q, visit_counts=visit_counts, fallback_action=fallback_action
        )
        if np.array_equal(new_actions, policy_actions):
            break
        policy_actions = new_actions

    return V, policy_actions, iterations


def make_greedy_policy(policy_actions: np.ndarray) -> Policy:
    """Wrap a table of discrete actions in a `Policy` that calls `encode`."""
    table = np.asarray(policy_actions, dtype=np.int64).copy()

    def policy(obs: np.ndarray) -> int:
        return int(table[encode(obs)])

    return policy


def train(
    scenario: str = "roundabout",
    reward: dict | None = None,
    step_budget: int = 5_000,
    seed: int = 0,
    *,
    method: str = "vi",
    gamma: float = DEFAULT_GAMMA,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> Policy:
    """Estimate the tabular MDP within `step_budget` env steps and solve it.

    Parameters
    ----------
    scenario:
        Scenario name accepted by `make_env` (default `"roundabout"`).
    reward:
        Optional reward-weight override dictionary passed to `make_env`.
    step_budget:
        Exact number of `env.step` calls allowed for collecting samples.
    seed:
        Random seed controlling both action sampling and episode resets.
    method:
        Planning solver: `"vi"` (Value Iteration) or `"pi"` (Policy Iteration).
    gamma:
        Discount factor in `[0, 1)`.
    """
    if method not in ("vi", "pi"):
        raise ValueError(f"Unknown method {method!r}. Expected 'vi' or 'pi'.")

    env = make_env(scenario=scenario, reward=reward)
    try:
        model = estimate_model(env, step_budget=step_budget, seed=seed)
    finally:
        env.close()

    if method == "vi":
        _V, policy_actions, _iters = value_iteration(
            model.P,
            model.R,
            gamma=gamma,
            tol=tol,
            max_iter=max_iter,
            visit_counts=model.visit_counts,
        )
    else:
        _V, policy_actions, _iters = policy_iteration(
            model.P,
            model.R,
            gamma=gamma,
            max_iter=min(max_iter, 1_000),
            visit_counts=model.visit_counts,
        )

    return make_greedy_policy(policy_actions)
