# Gap acceptance at an unsignalized roundabout

CSE 150A/250A final project (Team 28). An agent has to enter and drive through a busy roundabout without crashing and without freezing at the entry. We treat that as an MDP and compare two tabular methods on HighwayEnv's `roundabout-v1`:

- **Model-based:** estimate P(s'|s, a) from rollouts, then solve it with value iteration and policy iteration.
- **Model-free:** tabular Q-learning, which never builds that transition matrix.

The submitted proposal is `Team 28 CSE150A_250A Project Proposal - Google Docs.pdf`. It also names `highway-v0`. The first version of this project covers only the roundabout. The highway is a stretch goal if time allows, and `make_env("highway")` still works for when we get there.

## What the code does today

The simulator runs. Policies do not. The roundabout returns a `(5, 5)` kinematics observation (the agent's car plus four neighbors; features `presence`, `x`, `y`, `vx`, `vy`) and takes the actions `{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}`.

That observation is about to change. Sub-issue 1.1 switches to neighbor positions measured from the agent's car, in meters and meters per second, and adds the agent's heading. The observation becomes `(5, 7)`. Do not hard-code the shape; read it from `env.observation_space`.

A policy is a function from that observation to an action. Run it through `av_gap.rollout.run_episode` so return, episode length, and crash rate are computed the same way for every method. `scripts/random_baseline.py` is the baseline a learned policy should beat.

Use `roundabout-v1`, not `roundabout-v0`. The older version is deprecated: background cars miss neighbors across road segments and crash too often.

## Repo structure

```
pyproject.toml          highway-env==1.12.1; pip install -e . installs av_gap
.gitignore              leaves out .venv, the rounD zip, data/, and results/
README.md
tasks.md                short summary of the five work areas (details live in GitHub issues)
testEnv.py              one-step check that the roundabout resets and steps

av_gap/                 shared library. New learning code should import this
  envs.py               make_env(scenario="roundabout", reward=...). Reward keys
                        override collision_reward, high_speed_reward,
                        right_lane_reward, and lane_change_reward.
  rollout.py            random_policy and run_episode

scripts/                command-line entry points
  smoke_test.py         same check as testEnv.py
  random_baseline.py    five random roundabout episodes
```

Learned policies, the discretized state grid, and transition counts are not in the repo yet. Put them under `av_gap/` and call them from `scripts/`, using `run_episode` for evaluation.

The rounD zip (`round-dataset-v1.1.zip`) stays on your machine. It is licensed and too large for GitHub. It is used to calibrate how the roundabout's background cars follow and how fast they drive. Sub-issue 4.3 saves the measured values as constants in `av_gap/traffic.py`, so teammates without the zip can still run calibrated traffic. Do not commit the zip.

## Shared contract

Every learner and every experiment meets at these names. Change them only with the whole group's agreement, because the model-based and Q-learning code must stay interchangeable for the comparison to be fair.

- `av_gap.discretize.encode(obs) -> int` turns the observation into a bin number in `[0, n_states)`. The same observation always gives the same bin.
- `av_gap.discretize.n_states` is the total number of bins, used to size every table.
- `train(scenario, reward, step_budget, seed) -> policy` lives in both `av_gap/model_based.py` and `av_gap/q_learning.py`. Pass `"roundabout"` as the scenario. It calls `env.step` exactly `step_budget` times, and returns a policy that takes an observation and returns an action index from 0 to 4.
- `av_gap.rollout.run_episode(env, policy, seed) -> dict` scores any policy. Every number in the results comes from it.
- `make_env("roundabout", calibrated_traffic=True)`, added in sub-issue 4.3, builds the roundabout with background cars calibrated from rounD. Leaving the flag off gives HighwayEnv's default drivers.

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

The frozen-robot effect is the collision weight overpowering the speed weight: the agent learns that waiting at the entry forever is safest. Sweep `collision_reward` against `high_speed_reward` via `make_env(..., reward=...)` and report crash rate against progress at each ratio. Keep the proposal's 500-seed safety numbers (crash rate, minimum time-to-collision) for the final policies.

## Next work

1. Pin the observation config and discretize it into a shared state grid.
2. Estimate the transition model and solve it with value iteration and policy iteration.
3. Train tabular Q-learning against that same step budget.
4. Calibrate the roundabout's background traffic from rounD.
5. Run the learning curves, the reward-weight sweep, and the 500-drive table.
6. If time allows, repeat on `highway-v0`.
