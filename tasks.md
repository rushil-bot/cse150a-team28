Split the work into five pieces that meet at two shared functions. `encode(obs) -> int` turns the `(5, 5)` observation into a state index. `train(scenario, reward, step_budget, seed) -> policy` returns a function from observation to an action. `run_episode` in `av_gap/rollout.py` already scores any such policy.

## 1. State discretization

Own `av_gap/discretize.py`. Turn each kinematics row into a few bins the proposal already names: gap to the conflict or lead vehicle, relative speed, and how crowded the nearby lanes are. Drop empty neighbor rows (`presence == 0`) into one dedicated bin.

Done when `encode` is deterministic, `n_states` is small enough for a table (thousands, not millions), and the same call is used for highway and roundabout. Publish the function signature on the first day so tasks 2 and 3 can call a stub while the bins are still being tuned.

## 2. Model-based learning

Own `av_gap/model_based.py`. Roll out the discrete actions, count visits to `(s, a, s')`, and turn those counts into `P(s'|s, a)` and a reward table. Solve that table with value iteration and policy iteration. Every environment step spent filling the counts counts toward `step_budget`. The planning passes do not.

Done when `train(...)` returns a greedy policy that calls `encode`, and it can run on both scenarios with a reward-weight override.

## 3. Q-learning

Own `av_gap/q_learning.py`. Tabular Q-learning with ε-greedy exploration, using the same `encode`, the same five actions, and the same `train(...)` signature. Decay ε inside the step budget. Do not build a transition matrix.

Done when a finished policy is greedy over Q and the number of `env.step` calls it reports equals the budget it was given.

## 4. Roundabout traffic calibration

Own `av_gap/traffic.py`. This task never trains a policy. From the local rounD zip, measure median following gap, jam distance, and typical speed, then map those onto `TIME_WANTED`, `DISTANCE_WANTED`, and the roundabout target speeds. Leave the zip uncommitted. `highway-v0` stays on the textbook driver.

Done when `make_env("roundabout", traffic=...)` builds cars with those measured parameters, and a short note records the medians and how they were computed. A sampled distribution per car can follow the single median values.

## 5. Experiments and baselines

Own `scripts/evaluate.py` and extend `run_episode` with minimum time-to-collision. Call task 2 and task 3 with the same list of step budgets and plot return and crash rate against steps. On the same axes, include the random policy already in `scripts/random_baseline.py` and an ego car driven by HighwayEnv's own IDM driver. Sweep `collision_reward` against `high_speed_reward`. Run the final policies for 500 seeds and report crash rate, minimum time-to-collision, and progress.

Done when one command reproduces the learning curves, the weight sweep, and the 500-seed table for both scenarios.

Tasks 4 and 5 can start immediately. Tasks 2 and 3 start as soon as the `encode` stub exists. The roundabout numbers in the final table wait on task 4; the highway numbers do not.