# AntMaze Deep Reinforcement Learning

A beginner-friendly, research-organized Gymnasium Robotics AntMaze codebase using Stable-Baselines3.

This is a fresh DRL project, not tabular Q-learning. AntMaze has continuous actions and dictionary observations, so all algorithms use `MultiInputPolicy`.

## Project Structure

```text
.
|-- configs/
|-- src/
|-- checkpoints-<algorithm>-<version>/
|-- logs/
`-- results-<algorithm>-<version>/
```

Configs use:

```yaml
algorithm: PPO
version: v1
result_dir: auto
checkpoint_dir: auto
tensorboard_log: auto
monitor_dir: auto
final_model_name: auto
```

`auto` expands to names like:

```text
results-ppo-v1
checkpoints-ppo-v1
logs/tensorboard-ppo-v1
logs/monitor-ppo-v1
ppo_v1_final.zip
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Quick check:

```bash
python -c "import gymnasium, gymnasium_robotics, stable_baselines3, mujoco; print('AntMaze stack OK')"
```

## Phases

| Phase | Config | Result folder | Final model |
|---|---|---|---|
| 1 PPO UMaze Dense | `configs/ppo_antmaze_umaze_dense.yaml` | `results-ppo-v1` | `checkpoints-ppo-v1/ppo_v1_final.zip` |
| 2 SAC UMaze Dense | `configs/sac_antmaze_umaze_dense.yaml` | `results-sac-v1` | `checkpoints-sac-v1/sac_v1_final.zip` |
| 3 PPO Medium Dense | `configs/ppo_antmaze_medium_dense.yaml` | `results-ppo-v2` | `checkpoints-ppo-v2/ppo_v2_final.zip` |
| 3 SAC Medium Dense | `configs/sac_antmaze_medium_dense.yaml` | `results-sac-v2` | `checkpoints-sac-v2/sac_v2_final.zip` |
| 4 SAC UMaze Sparse | `configs/sac_antmaze_umaze_sparse.yaml` | `results-sac-v3` | `checkpoints-sac-v3/sac_v3_final.zip` |
| 4 TD3 UMaze Sparse | `configs/td3_antmaze_umaze_sparse.yaml` | `results-td3-v3` | `checkpoints-td3-v3/td3_v3_final.zip` |
| 5 SAC+HER UMaze Sparse | `configs/sac_antmaze_umaze_sparse_her.yaml` | `results-sac-v4` | `checkpoints-sac-v4/sac_v4_final.zip` |

## Commands

Phase 1:

```bash
python -m src.train --config configs/ppo_antmaze_umaze_dense.yaml
python -m src.evaluate --config configs/ppo_antmaze_umaze_dense.yaml --model checkpoints-ppo-v1/ppo_v1_final.zip --episodes 50
python -m src.visualize --config configs/ppo_antmaze_umaze_dense.yaml --model checkpoints-ppo-v1/ppo_v1_final.zip
```

Phase 2:

```bash
python -m src.train --config configs/sac_antmaze_umaze_dense.yaml
python -m src.evaluate --config configs/sac_antmaze_umaze_dense.yaml --model checkpoints-sac-v1/sac_v1_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_umaze_dense.yaml --model checkpoints-sac-v1/sac_v1_final.zip
```

Phase 3:

```bash
python -m src.train --config configs/ppo_antmaze_medium_dense.yaml
python -m src.evaluate --config configs/ppo_antmaze_medium_dense.yaml --model checkpoints-ppo-v2/ppo_v2_final.zip --episodes 50
python -m src.visualize --config configs/ppo_antmaze_medium_dense.yaml --model checkpoints-ppo-v2/ppo_v2_final.zip

python -m src.train --config configs/sac_antmaze_medium_dense.yaml
python -m src.evaluate --config configs/sac_antmaze_medium_dense.yaml --model checkpoints-sac-v2/sac_v2_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_medium_dense.yaml --model checkpoints-sac-v2/sac_v2_final.zip
```

Phase 4:

```bash
python -m src.train --config configs/sac_antmaze_umaze_sparse.yaml
python -m src.evaluate --config configs/sac_antmaze_umaze_sparse.yaml --model checkpoints-sac-v3/sac_v3_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_umaze_sparse.yaml --model checkpoints-sac-v3/sac_v3_final.zip

python -m src.train --config configs/td3_antmaze_umaze_sparse.yaml
python -m src.evaluate --config configs/td3_antmaze_umaze_sparse.yaml --model checkpoints-td3-v3/td3_v3_final.zip --episodes 50
python -m src.visualize --config configs/td3_antmaze_umaze_sparse.yaml --model checkpoints-td3-v3/td3_v3_final.zip
```

Phase 5:

```bash
python -m src.train --config configs/sac_antmaze_umaze_sparse_her.yaml
python -m src.evaluate --config configs/sac_antmaze_umaze_sparse_her.yaml --model checkpoints-sac-v4/sac_v4_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_umaze_sparse_her.yaml --model checkpoints-sac-v4/sac_v4_final.zip
```

Resume any run:

```bash
python -m src.train \
  --config configs/sac_antmaze_umaze_dense.yaml \
  --resume checkpoints-sac-v1/sac_v1_final_step_1000000_steps.zip
```

## Metrics

Evaluation writes:

```text
evaluation_results.csv
evaluation_rewards.png
evaluation_distances.png
best_rollout_trace.csv
best_rollout_distance_over_time.png
best_rollout_xy_trajectory.png
```

Compare models with:

- success rate
- average minimum distance
- average final distance
- unhealthy steps and unhealthy penalty
- distance-to-goal over time
- xy trajectory toward the goal

## Notes

`penalize_unhealthy` adds a negative reward every timestep the ant is unhealthy or flipped. With `terminate_on_unhealthy: false`, the episode continues and the penalty grows the longer the ant stays upside down.

HER is configured as a replay buffer for SAC in `sac_antmaze_umaze_sparse_her.yaml`; it is not a separate algorithm.
