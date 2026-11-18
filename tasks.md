The work splits into five pieces that meet at a few shared functions listed in the README's "Shared contract." The detailed sub-tasks live in the GitHub issues; this file is the one-page overview. The first version covers only the roundabout (`roundabout-v1`). The highway is a stretch goal.

## 1. State discretization

Own `av_gap/discretize.py`. First pin one observation config in `make_env`: neighbor positions measured from the agent's car, in real units, plus the agent's heading, giving a `(5, 7)` observation. Then turn that into a few bins the proposal already names: gap to the conflicting or leading car, relative speed, and how crowded the ring is near the entry. Empty neighbor rows (`presence == 0`) go into their own bin.

Done when `encode` is deterministic and `n_states` is small enough for a table (thousands, not millions). Publish the function signature on the first day so tasks 2 and 3 can call a stub while the bins are still being tuned.

## 2. Model-based learning

Own `av_gap/model_based.py`. Roll out the discrete actions, count visits to `(s, a, s')`, and turn those counts into `P(s'|s, a)` and a reward table. Solve that table with value iteration and policy iteration. Every environment step spent filling the counts counts toward `step_budget`. The planning passes do not.

Done when `train(...)` returns a greedy policy that calls `encode`, and it runs on the roundabout with a reward-weight override.

## 3. Q-learning

Own `av_gap/q_learning.py`. Tabular Q-learning with ε-greedy exploration, using the same `encode`, the same five actions, and the same `train(...)` signature. Decay ε inside the step budget. Do not build a transition matrix.

Done when a finished policy is greedy over Q and the number of `env.step` calls it reports equals the budget it was given.

## 4. Roundabout traffic calibration

Own `av_gap/traffic.py`. This task never trains a policy. From the local rounD zip, measure the following time gap, jam distance, and typical speed, and save them (with a few percentiles) as constants. A `CalibratedVehicle` subclass applies them to the background cars. Leave the zip uncommitted.

Done when `make_env("roundabout", calibrated_traffic=True)` builds cars with those measured values, and a short note records the medians and how they were computed. Drawing different values per car can follow the single median values.

## 5. Experiments and baselines

Own `scripts/evaluate.py` and extend `run_episode` with minimum time-to-collision. Call task 2 and task 3 with the same list of step budgets and plot return and crash rate against steps. On the same axes, include the random policy already in `scripts/random_baseline.py` and an agent car driven by HighwayEnv's own IDM driver. Sweep `collision_reward` against `high_speed_reward`. Run the final policies for 500 seeds and report crash rate, minimum time-to-collision, and progress, with default and calibrated traffic.

Done when one command reproduces the learning curves, the weight sweep, and the 500-seed table on the roundabout.

Tasks 1, 4, and 5's baselines can start immediately. Tasks 2 and 3 start as soon as the `encode` stub exists. The calibrated-traffic rows of the final table wait on task 4; the default-traffic rows do not.
