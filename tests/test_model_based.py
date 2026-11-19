"""Tests for model-based tabular RL (Value Iteration and Policy Iteration)."""

import numpy as np
import pytest

from av_gap.discretize import encode, n_states
from av_gap.envs import make_env
from av_gap.model_based import (
    IDLE_ACTION,
    build_model_from_counts,
    estimate_model,
    policy_iteration,
    train,
    value_iteration,
)
from av_gap.rollout import run_episode


def test_vi_and_pi_agree_on_synthetic_mdp():
    """VI and PI must find the same optimal V* and policy on a stochastic MDP."""
    # 3 states, 2 actions:
    # State 0: action 0 stays in 0 (r=0.2), action 1 goes to state 1 with p=0.8 or crashes (terminal) with p=0.2 (r=0.0)
    # State 1: action 0 goes to state 2 (r=1.0), action 1 crashes (r=-1.0)
    # State 2: absorbing reward state, action 0 stays in 2 (r=1.0), action 1 goes to 0 (r=0.0)
    P = np.zeros((3, 2, 3), dtype=np.float64)
    R = np.zeros((3, 2), dtype=np.float64)

    P[0, 0, 0] = 1.0
    R[0, 0] = 0.2
    P[0, 1, 1] = 0.8  # 0.2 mass goes to absorbing terminal state (value 0)
    R[0, 1] = 0.0

    P[1, 0, 2] = 1.0
    R[1, 0] = 1.0
    # P[1, 1, :] = 0.0 (certain crash)
    R[1, 1] = -1.0

    P[2, 0, 2] = 1.0
    R[2, 0] = 1.0
    P[2, 1, 0] = 1.0
    R[2, 1] = 0.0

    gamma = 0.9
    V_vi, pi_vi, iters_vi = value_iteration(P, R, gamma=gamma, tol=1e-10)
    V_pi, pi_pi, iters_pi = policy_iteration(P, R, gamma=gamma)

    assert iters_vi > 0
    assert iters_pi > 0
    assert np.allclose(V_vi, V_pi, atol=1e-8)
    assert np.array_equal(pi_vi, pi_pi)
    # Optimal policy: take action 1 from state 0 (reaches high-reward state 2),
    # action 0 from state 1, action 0 from state 2.
    assert pi_pi.tolist() == [1, 0, 0]
    assert V_pi[2] == pytest.approx(1.0 / (1.0 - gamma), rel=1e-6)
    assert V_pi[1] == pytest.approx(1.0 + gamma * V_pi[2], rel=1e-6)
    assert V_pi[0] == pytest.approx(0.8 * gamma * V_pi[1], rel=1e-6)


def test_unvisited_states_fall_back_to_idle():
    P = np.zeros((4, 5, 4), dtype=np.float64)
    R = np.zeros((4, 5), dtype=np.float64)
    visit_counts = np.zeros((4, 5), dtype=np.int64)

    # Only state 0 is visited; action 3 is best there.
    visit_counts[0, 3] = 10
    P[0, 3, 0] = 1.0
    R[0, 3] = 1.0

    _V_vi, pi_vi, _ = value_iteration(P, R, visit_counts=visit_counts)
    _V_pi, pi_pi, _ = policy_iteration(P, R, visit_counts=visit_counts)

    assert pi_vi[0] == 3
    assert pi_pi[0] == 3
    assert list(pi_vi[1:]) == [IDLE_ACTION, IDLE_ACTION, IDLE_ACTION]
    assert list(pi_pi[1:]) == [IDLE_ACTION, IDLE_ACTION, IDLE_ACTION]


def test_build_model_from_counts_handles_terminal_transitions():
    visit_counts = np.zeros((2, 2), dtype=np.int64)
    transition_counts = np.zeros((2, 2, 2), dtype=np.int64)
    terminal_counts = np.zeros((2, 2), dtype=np.int64)
    reward_sum = np.zeros((2, 2), dtype=np.float64)

    # (s=0, a=0): 4 visits -> 3 to s'=1, 1 crash; total reward = 2.0
    visit_counts[0, 0] = 4
    transition_counts[0, 0, 1] = 3
    terminal_counts[0, 0] = 1
    reward_sum[0, 0] = 2.0

    model = build_model_from_counts(
        visit_counts, transition_counts, terminal_counts, reward_sum
    )
    assert model.P[0, 0, 1] == pytest.approx(0.75)
    assert model.P[0, 0].sum() == pytest.approx(0.75)
    assert model.R[0, 0] == pytest.approx(0.5)
    # Unvisited (s=1, a=1) stays zero
    assert model.P[1, 1].sum() == 0.0
    assert model.R[1, 1] == 0.0


def test_estimate_model_uses_exact_step_budget():
    env = make_env("roundabout")
    step_calls = 0
    orig_step = env.step

    def counting_step(action):
        nonlocal step_calls
        step_calls += 1
        return orig_step(action)

    env.step = counting_step
    try:
        budget = 25
        model = estimate_model(env, step_budget=budget, seed=42)
    finally:
        env.close()

    assert step_calls == budget
    assert model.steps_collected == budget
    assert int(model.visit_counts.sum()) == budget
    assert int(model.transition_counts.sum() + model.terminal_counts.sum()) == budget
    assert model.P.shape == (n_states, 5, n_states)
    assert model.R.shape == (n_states, 5)
    # Row sums of P never exceed 1
    assert np.all(model.P.sum(axis=2) <= 1.0 + 1e-12)


@pytest.mark.parametrize("method", ["vi", "pi"])
def test_train_contract_and_reward_override(method):
    reward_override = {"collision_reward": -3.0, "high_speed_reward": 0.4}
    policy = train(
        scenario="roundabout",
        reward=reward_override,
        step_budget=30,
        seed=7,
        method=method,
    )

    env = make_env("roundabout", reward=reward_override)
    try:
        obs, _info = env.reset(seed=0)
        action = policy(obs)
        assert isinstance(action, int)
        assert 0 <= action < 5

        result = run_episode(env, policy, seed=1)
        assert set(result.keys()) == {"return", "length", "crashed"}
        assert result["length"] >= 1
    finally:
        env.close()


def test_train_is_deterministic_for_same_seed():
    p1 = train("roundabout", None, step_budget=20, seed=123, method="vi")
    p2 = train("roundabout", None, step_budget=20, seed=123, method="vi")

    env = make_env("roundabout")
    try:
        obs, _info = env.reset(seed=0)
        assert p1(obs) == p2(obs)
    finally:
        env.close()


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        train("roundabout", None, step_budget=10, seed=0, method="q_learning")
    with pytest.raises(ValueError):
        value_iteration(np.zeros((2, 2, 2)), np.zeros((2, 2)), gamma=1.0)
    with pytest.raises(ValueError):
        policy_iteration(np.zeros((2, 2, 2)), np.zeros((2, 2)), gamma=-0.1)
