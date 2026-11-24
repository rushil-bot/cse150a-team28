# Gap acceptance at an unsignalized roundabout

CSE 150A/250A final project (Team 28). An agent has to enter and drive through a busy roundabout without crashing and without freezing at the entry. We treat this as an MDP and compare two tabular methods on HighwayEnv's `roundabout-v1`:

- **Model-based:** estimate P(s'|s, a) from rollouts, then solve it with value iteration and policy iteration.
- **Model-free:** tabular Q-learning, which never builds a transition model.

Both are compared at equal environment-step budgets against a random policy and HighwayEnv's IDM driver. They are tested with default traffic and with background traffic calibrated from the rounD dataset.

## Setup

```powershell
git clone https://github.com/rushil-bot/cse150a-team28.git
cd cse150a-team28
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required.

## Usage

```powershell
python scripts/train_model_based.py             # train VI and evaluate it on 20 seeds
python scripts/train_model_based.py --method pi # same, with policy iteration
python scripts/watch.py                         # watch episodes in a window, with state and action printed each step
python scripts/evaluate.py                      # random vs. IDM baselines
python scripts/evaluate.py table                # 500-drive results table
python scripts/evaluate.py sweep                # collision weight vs. progress (frozen-robot effect)
python scripts/evaluate.py curves               # learning curves at equal step budgets
python scripts/plot.py                          # redraw figures and tables from results/*.csv
python -m pytest tests -q                       # run the tests
```

Results are saved to `results/`, which is not committed. Each experiment script accepts `--help`. To check that a command works before a full run, give it small settings first, for example `table --budgets 200 --episodes 5`.

## How it works

- **Environment** (`av_gap/envs.py`): `make_env(scenario, reward=..., calibrated_traffic=...)` returns a `(5, 7)` kinematics observation of the agent and its four nearest cars. The actions are `LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER`. Use `reward` to override HighwayEnv's reward weights.
- **States** (`av_gap/discretize.py`): `encode(obs)` maps an observation to one of 135 discrete states. A state combines four binned quantities: agent speed (3 bins), phase (approaching, on the ring, or exiting), the gap to the most dangerous car (5 bins), and how fast that gap is closing (3 bins). `describe(state)` turns a state back into readable text.
- **Learners** (`av_gap/model_based.py`, `av_gap/q_learning.py`): both learners expose `train(scenario, reward, step_budget, seed) -> policy` and call `env.step` exactly `step_budget` times. Collecting samples for the model counts toward the budget; planning does not.
- **Traffic** (`av_gap/traffic.py`): jam gap, time gap, and target speed measured from rounD are stored as constants. Each `CalibratedVehicle` draws its own values from that measured spread.
- **Evaluation** (`av_gap/rollout.py`, `av_gap/evaluation.py`): `run_episode` scores every policy the same way: return, crash, minimum time-to-collision, distance, and speed. Evaluation seeds start at 10,000 and training seeds at 20,000, so the two never overlap.

## Repo structure

```
av_gap/      library: envs, discretize, model_based, q_learning, traffic, rollout, evaluation, plots
scripts/     command-line entry points (see Usage)
tests/       pytest suite, also run by CI in .github/workflows/
```

The rounD zip (`round-dataset-v1.1.zip`) is licensed, so it is not in the repo. You only need it to re-measure the traffic constants in `av_gap/traffic.py`.
