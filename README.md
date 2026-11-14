# Gap acceptance in highway and roundabout traffic

CSE 150A/250A final project (Team 28). An agent has to make progress through dense traffic without freezing: highway driving and an unsignalized roundabout. We treat that as an MDP and compare two tabular methods on the same HighwayEnv tasks:

- **Model-based:** estimate P(s'|s, a) from rollouts, then solve it with value iteration and policy iteration.
- **Model-free:** tabular Q-learning, which never builds that transition matrix.

The submitted proposal is `Team 28 CSE150A_250A Project Proposal - Google Docs.pdf`. This README is the working plan for the code.

## What the code does today

The simulator runs. Policies do not. Both scenarios use a `(5, 5)` kinematics observation (ego plus four neighbors; features `presence`, `x`, `y`, `vx`, `vy`) and the actions `{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}`.

A policy is a function from that observation to an action. Run it through `av_gap.rollout.run_episode` so return, episode length, and crash rate are computed the same way for every method. `scripts/random_baseline.py` is the baseline a learned policy should beat.

`roundabout-v1` is the roundabout to use. `roundabout-v0` is deprecated: background cars miss neighbors across road segments and crash too often. The highway task is `highway-v0`.

## Repo structure

```
pyproject.toml          highway-env==1.12.1; pip install -e . installs av_gap
.gitignore              leaves out .venv, the rounD zip, data/, and results/
README.md
testEnv.py              one-step check that both scenarios reset and step

av_gap/                 shared library. New learning code should import this
  envs.py               make_env(scenario, reward=...). Reward keys override
                        collision_reward, high_speed_reward, right_lane_reward,
                        and lane_change_reward.
  rollout.py            random_policy and run_episode

scripts/                command-line entry points
  smoke_test.py         same check as testEnv.py
  random_baseline.py    five random episodes per scenario
```

Learned policies, the discretized state grid, and transition counts are not in the repo yet. Put them under `av_gap/` and call them from `scripts/`, using `run_episode` for evaluation.

The rounD zip (`round-dataset-v1.1.zip`) stays on your machine. It is licensed and too large for GitHub. Later it is used to calibrate background-vehicle headway and yielding. Do not commit it.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python testEnv.py
python scripts/random_baseline.py
```

## Evaluation notes

The proposal is already submitted. These two points came back from the TA review, so the experiments should follow them.

Compare value iteration, policy iteration, and Q-learning at the **same number of environment steps**. Plot evaluation return and crash rate against steps consumed. Samples used to estimate P(s'|s, a) count. Planning compute does not. Q-learning updates count the same way. A single final score after unequal training is not a comparison.

The frozen-robot effect is the collision weight overpowering the speed weight. Sweep `collision_reward` against `high_speed_reward` via `make_env(..., reward=...)` and report crash rate against progress at each ratio. Keep the proposal's 500-seed safety numbers (crash rate, minimum time-to-collision) for the final policies.

## Next work

1. Discretize the kinematics observation into a shared state grid.
2. Estimate the transition model and solve it with value iteration and policy iteration.
3. Train tabular Q-learning against that same step budget.
4. Run the reward-weight sweep on both.
