# Gap acceptance at an unsignalized roundabout

CSE 150A/250A final project (Team 28). An agent has to enter and drive through a busy roundabout without crashing and without freezing at the entry. We treat that as an MDP and compare two tabular methods on HighwayEnv's `roundabout-v1`:

- **Model-based:** estimate P(s'|s, a) from rollouts, then solve it with value iteration and policy iteration.
- **Model-free:** tabular Q-learning, which never builds that transition matrix.

The submitted proposal is `Team 28 CSE150A_250A Project Proposal - Google Docs.pdf`. It also names `highway-v0`. The first version of this project covers only the roundabout. The highway is a stretch goal if time allows, and `make_env("highway")` still works for when we get there.

## Quickstart

```powershell
git clone https://github.com/rushil-bot/cse150a-team28.git
cd cse150a-team28
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. Python 3.10 or newer is required.

Then try these:

```powershell
python scripts/watch.py            # watch the agent drive, in a window
python scripts/watch.py 10         # watch 10 episodes instead of 3
python testEnv.py                  # one step; prints the observation format
python scripts/random_baseline.py  # 5 episodes; prints mean return and crash rate
python scripts/evaluate.py         # random vs. IDM baseline, 20 episodes each
python -m pytest tests -q          # run the tests
```

Start with `scripts/watch.py`. The agent is the green car waiting at the south entry; the blue cars are circulating traffic. Every car is picking random actions right now, so expect crashes. Each step prints the state and the action taken, so a crash can be explained rather than just watched:

```
   4  medium, on ring, gap >=40 m, closing fast             IDLE
   5  medium, on ring, gap <10 m, closing medium            SLOWER
  crashed after 6 decisions, return 4.50
```

Episodes are seeded, so any crash can be replayed. Once a learner exists, swap `random_policy` for its trained policy in that script to watch what it actually does.

## What the code does today

The simulator runs. Policies do not. `make_env` returns a `(5, 7)` observation: row 0 is the agent's car, and rows 1–4 are its four nearest neighbors. Columns are `presence`, `x`, `y`, `vx`, `vy`, `cos_h`, `sin_h`, in meters and m/s. Row 0 is in map coordinates. For neighbors, position and velocity are offsets from the agent's car, and `cos_h`, `sin_h` are the neighbor's own heading. Use `av_gap.envs.COL` to find a column by name. Actions are `{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}`.

`av_gap.discretize.encode` maps an observation to one of 135 states. See "States and bins" below for what that means and why we do it.

A policy is a function from that observation to an action. Run it through `av_gap.rollout.run_episode` so return, episode length, and crash rate are computed the same way for every method. `scripts/random_baseline.py` is the baseline a learned policy should beat.

Use `roundabout-v1`, not `roundabout-v0`. The older version is deprecated: background cars miss neighbors across road segments and crash too often.

## States and bins

Both of our methods learn by filling in a table with one row per situation. Q-learning stores how good each action looks; the model-based method stores counts of what happened next. Either way the table needs a finite list of situations.

The simulator does not give us one. It reports things like "the nearest car is 13.72 m away, closing at 4.18 m/s," and those numbers are continuous, so the same values essentially never come up twice. Every row would be visited once and nothing would be learned.

A **bin** is a bucket of nearby values, used so that similar situations count as the same situation. For the gap to the nearest car we use five:

| Measured gap | Bin |
|---|---|
| nothing closing in | "no car closing" |
| 3.1 m or 8.9 m | "under 10 m" |
| 13.7 m | "10–20 m" |
| 31.0 m | "20–40 m" |
| 55.0 m | "40 m or more" |

Whatever the agent learns at 3.1 m now also applies at 8.9 m. The cutoffs are the **bin edges**, and picking them is a trade-off. Too many bins and the table is huge with most rows rarely visited. Too few and genuinely different situations get lumped together: with one "under 40 m" bucket, a car 2 m away looks the same as one 39 m away.

We bin four quantities and combine them:

| Quantity | Bins |
|---|---|
| Agent speed | stopped, medium, fast |
| Phase | approaching, on the ring, exiting |
| Gap to the most dangerous car | the five above |
| How fast that gap is closing | slow, medium, fast |

That gives 3 × 3 × 5 × 3 = 135 combinations. `encode` returns which one the current observation falls into, as a single integer, and `n_states` is 135. Each learner's table then has 135 rows and 5 columns, one per action.

The single integer is for the learners' convenience: they treat states as unrelated labels and never assume state 37 resembles state 38. For humans, `describe(state)` converts it back, for example `"medium, approaching, gap 10-20 m, closing fast"`. Run `python scripts/bin_coverage.py` to see which states actually come up and whether any bin is swallowing most of the steps.

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
  discretize.py         encode, n_states, describe, nearest_threat
  rollout.py            run_episode, time_to_collision, and the random and
                        IDM baselines (random_policy, IDMDriven + idm_policy)

scripts/                command-line entry points
  smoke_test.py         same check as testEnv.py
  random_baseline.py    five random roundabout episodes
  evaluate.py           random vs. IDM baseline: return, crash rate, min TTC
  watch.py              play episodes in a window, printing state and action
  bin_coverage.py       how often each state bin is used; for tuning the edges

tests/                  run with `python -m pytest tests -q`
  test_envs.py          observation format and reward overrides
  test_discretize.py    the `encode` contract and what each bin means
  test_rollout.py       min_ttc, time_to_collision, and the IDM baseline

.github/workflows/      CI; runs the tests on every pull request
```

Learned policies, the discretized state grid, and transition counts are not in the repo yet. Put them under `av_gap/` and call them from `scripts/`, using `run_episode` for evaluation.

The rounD zip (`round-dataset-v1.1.zip`) stays on your machine. It is licensed and too large for GitHub. It is used to calibrate how the roundabout's background cars follow and how fast they drive. Sub-issue 4.3 saves the measured values as constants in `av_gap/traffic.py`, so teammates without the zip can still run calibrated traffic. Do not commit the zip.

## Shared contract

Every learner and every experiment meets at these names. Change them only with the whole group's agreement, because the model-based and Q-learning code must stay interchangeable for the comparison to be fair.

- `av_gap.discretize.encode(obs) -> int` turns the observation into a bin number in `[0, n_states)`. The same observation always gives the same bin.
- `av_gap.discretize.n_states` is the total number of bins, used to size every table.
- `av_gap.discretize.describe(state) -> str` turns a state index back into readable text.
- `av_gap.discretize.nearest_threat(obs)` returns the soonest-contact neighbor as `(distance, closing speed, seconds)`. Issue 5.1 should use this for its safety measure rather than writing a second copy.
- `train(scenario, reward, step_budget, seed) -> policy` lives in both `av_gap/model_based.py` and `av_gap/q_learning.py`. Pass `"roundabout"` as the scenario. It calls `env.step` exactly `step_budget` times, and returns a policy that takes an observation and returns an action index from 0 to 4.
- `av_gap.rollout.run_episode(env, policy, seed) -> dict` scores any policy, returning `return`, `length`, `crashed`, and `min_ttc` (smallest time-to-collision in seconds, infinity if nothing ever closed in). Every number in the results comes from it.
- `make_env("roundabout", calibrated_traffic=True)`, added in sub-issue 4.3, builds the roundabout with background cars calibrated from rounD. Leaving the flag off gives HighwayEnv's default drivers.

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
