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
version: v1-standard
result_dir: auto
checkpoint_dir: auto
tensorboard_log: auto
monitor_dir: auto
final_model_name: auto
```

`auto` expands to names like:

```text
results-ppo-v1-standard
checkpoints-ppo-v1-standard
logs/tensorboard-ppo-v1-standard
logs/monitor-ppo-v1-standard
ppo_v1_standard_final.zip
```

Standard configs use the original Gymnasium Robotics reward only. Configs marked as shaped add explicit reward shaping terms and should be reported separately from standard baselines.

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
| 1 PPO UMaze Dense Shaped | `configs/ppo_antmaze_umaze_dense.yaml` | `results-ppo-v1-shaped` | `checkpoints-ppo-v1-shaped/ppo_v1_shaped_final.zip` |
| 2 SAC UMaze Dense | `configs/sac_antmaze_umaze_dense.yaml` | `results-sac-v1-standard` | `checkpoints-sac-v1-standard/sac_v1_standard_final.zip` |
| 3 PPO Medium Dense | `configs/ppo_antmaze_medium_dense.yaml` | `results-ppo-v2-standard` | `checkpoints-ppo-v2-standard/ppo_v2_standard_final.zip` |
| 3 SAC Medium Dense | `configs/sac_antmaze_medium_dense.yaml` | `results-sac-v2-standard` | `checkpoints-sac-v2-standard/sac_v2_standard_final.zip` |
| 4 SAC UMaze Sparse | `configs/sac_antmaze_umaze_sparse.yaml` | `results-sac-v3-standard` | `checkpoints-sac-v3-standard/sac_v3_standard_final.zip` |
| 4 TD3 UMaze Sparse | `configs/td3_antmaze_umaze_sparse.yaml` | `results-td3-v3-standard` | `checkpoints-td3-v3-standard/td3_v3_standard_final.zip` |
| 5 SAC+HER UMaze Sparse | `configs/sac_antmaze_umaze_sparse_her.yaml` | `results-sac-v4-standard` | `checkpoints-sac-v4-standard/sac_v4_standard_final.zip` |
| 5 SAC UMaze Dense Fine-tune | `configs/sac_antmaze_umaze_dense_finetune.yaml` | `results-sac-v5-umaze-dense-finetune` | `checkpoints-sac-v5-umaze-dense-finetune/sac_v5_umaze_dense_finetune_final.zip` |
| 5 SAC Medium Dense Fine-tune | `configs/sac_antmaze_medium_dense_finetune.yaml` | `results-sac-v5-medium-dense-finetune` | `checkpoints-sac-v5-medium-dense-finetune/sac_v5_medium_dense_finetune_final.zip` |
| 5 TD3 UMaze Dense Fine-tune | `configs/td3_antmaze_umaze_dense_finetune.yaml` | `results-td3-v5-umaze-dense-finetune` | `checkpoints-td3-v5-umaze-dense-finetune/td3_v5_umaze_dense_finetune_final.zip` |
| 5 TD3 Medium Dense Fine-tune | `configs/td3_antmaze_medium_dense_finetune.yaml` | `results-td3-v5-medium-dense-finetune` | `checkpoints-td3-v5-medium-dense-finetune/td3_v5_medium_dense_finetune_final.zip` |

## Commands

Phase 1:

```bash
python -m src.train --config configs/ppo_antmaze_umaze_dense.yaml
python -m src.evaluate --config configs/ppo_antmaze_umaze_dense.yaml --model checkpoints-ppo-v1-shaped/ppo_v1_shaped_final.zip --episodes 50
python -m src.visualize --config configs/ppo_antmaze_umaze_dense.yaml --model checkpoints-ppo-v1-shaped/ppo_v1_shaped_final.zip
```

Phase 2:

```bash
python -m src.train --config configs/sac_antmaze_umaze_dense.yaml
python -m src.evaluate --config configs/sac_antmaze_umaze_dense.yaml --model checkpoints-sac-v1-standard/sac_v1_standard_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_umaze_dense.yaml --model checkpoints-sac-v1-standard/sac_v1_standard_final.zip
```

Phase 3:

```bash
python -m src.train --config configs/ppo_antmaze_medium_dense.yaml
python -m src.evaluate --config configs/ppo_antmaze_medium_dense.yaml --model checkpoints-ppo-v2-standard/ppo_v2_standard_final.zip --episodes 50
python -m src.visualize --config configs/ppo_antmaze_medium_dense.yaml --model checkpoints-ppo-v2-standard/ppo_v2_standard_final.zip

python -m src.train --config configs/sac_antmaze_medium_dense.yaml
python -m src.evaluate --config configs/sac_antmaze_medium_dense.yaml --model checkpoints-sac-v2-standard/sac_v2_standard_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_medium_dense.yaml --model checkpoints-sac-v2-standard/sac_v2_standard_final.zip
```

Phase 4:

```bash
python -m src.train --config configs/sac_antmaze_umaze_sparse.yaml
python -m src.evaluate --config configs/sac_antmaze_umaze_sparse.yaml --model checkpoints-sac-v3-standard/sac_v3_standard_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_umaze_sparse.yaml --model checkpoints-sac-v3-standard/sac_v3_standard_final.zip

python -m src.train --config configs/td3_antmaze_umaze_sparse.yaml
python -m src.evaluate --config configs/td3_antmaze_umaze_sparse.yaml --model checkpoints-td3-v3-standard/td3_v3_standard_final.zip --episodes 50
python -m src.visualize --config configs/td3_antmaze_umaze_sparse.yaml --model checkpoints-td3-v3-standard/td3_v3_standard_final.zip
```

Phase 5:

```bash
python -m src.train --config configs/sac_antmaze_umaze_sparse_her.yaml
python -m src.evaluate --config configs/sac_antmaze_umaze_sparse_her.yaml --model checkpoints-sac-v4-standard/sac_v4_standard_final.zip --episodes 50
python -m src.visualize --config configs/sac_antmaze_umaze_sparse_her.yaml --model checkpoints-sac-v4-standard/sac_v4_standard_final.zip
```

Phase 5 dense fine-tuning from an existing SAC checkpoint:

```bash
python -m src.train \
  --config configs/sac_antmaze_umaze_dense_finetune.yaml \
  --init-weights checkpoints-sac-v4-standard/sac_v4_standard_final_step_900000_steps.zip

python -m src.evaluate \
  --config configs/sac_antmaze_umaze_dense_finetune.yaml \
  --model checkpoints-sac-v5-umaze-dense-finetune/sac_v5_umaze_dense_finetune_final.zip \
  --episodes 50

python -m src.visualize \
  --config configs/sac_antmaze_umaze_dense_finetune.yaml \
  --model checkpoints-sac-v5-umaze-dense-finetune/sac_v5_umaze_dense_finetune_final.zip
```

To fine-tune on Medium Dense instead:

```bash
python -m src.train \
  --config configs/sac_antmaze_medium_dense_finetune.yaml \
  --init-weights checkpoints-sac-v1-standard/sac_v1_standard_final.zip
```

TD3 dense fine-tuning follows the same idea. Start with UMaze Dense:

```bash
python -m src.train \
  --config configs/td3_antmaze_umaze_dense_finetune.yaml \
  --init-weights checkpoints-td3-v3-standard/td3_v3_standard_final.zip

python -m src.evaluate \
  --config configs/td3_antmaze_umaze_dense_finetune.yaml \
  --model checkpoints-td3-v5-umaze-dense-finetune/td3_v5_umaze_dense_finetune_final.zip \
  --episodes 50

python -m src.visualize \
  --config configs/td3_antmaze_umaze_dense_finetune.yaml \
  --model checkpoints-td3-v5-umaze-dense-finetune/td3_v5_umaze_dense_finetune_final.zip
```

Then move TD3 to Medium Dense:

```bash
python -m src.train \
  --config configs/td3_antmaze_medium_dense_finetune.yaml \
  --init-weights checkpoints-td3-v5-umaze-dense-finetune/td3_v5_umaze_dense_finetune_final.zip
```

Continue the exact same run:

```bash
python -m src.train \
  --config configs/sac_antmaze_umaze_dense.yaml \
  --resume checkpoints-sac-v1-standard/sac_v1_standard_final_step_1000000_steps.zip
```

Use `--resume` to continue the same algorithm settings from the saved `.zip`. Use `--init-weights` to copy the neural-network weights into a new run that uses the new config, environment, learning rate, and replay buffer settings.

## TensorBoard

View all training runs:

```bash
tensorboard --logdir logs
```

View only TensorBoard event logs:

```bash
tensorboard --logdir logs/tensorboard-sac-v1-standard
```

Common run-specific examples:

```bash
tensorboard --logdir logs/tensorboard-ppo-v2-standard
tensorboard --logdir logs/tensorboard-sac-v2-standard
tensorboard --logdir logs/tensorboard-sac-v4-standard
tensorboard --logdir logs/tensorboard-sac-v5-umaze-dense-finetune
tensorboard --logdir logs/tensorboard-td3-v5-umaze-dense-finetune
```

Then open the local URL TensorBoard prints, usually:

```text
http://localhost:6006/
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
rollouts/*_topdown_map_trajectory.png
```

Compare models with:

- success rate
- average minimum distance
- average final distance
- distance-to-goal over time
- xy trajectory toward the goal
- top-down map trajectory through the maze

Generate report-ready comparison assets:

```bash
python -m src.report_assets \
  --run "PPO Medium Dense=results-ppo-v2-standard" \
  --run "SAC Medium Dense=results-sac-v2-standard" \
  --output-dir results-report-assets
```

This writes:

```text
comparison_table.csv
comparison_table.md
comparison_table.png
training_curves.png
```

Use the comparison table for average return, success rate, final/minimum distance, training steps, and an estimated convergence step. Use `training_curves.png` as the report training-curve figure.





## Notes

Dense AntMaze rewards can be extremely small when the ant is far from the goal. Sparse AntMaze gives no useful reward until success, so plain SAC/TD3 may appear to receive all zeros during early exploration.

The shaped PPO UMaze config adds a small flip/tilt penalty, a small vertical-motion penalty, and a goal-progress reward. Treat it as reward shaping, not as the standard AntMaze baseline.

HER is configured as a replay buffer for SAC in `sac_antmaze_umaze_sparse_her.yaml`; it is not a separate algorithm.
