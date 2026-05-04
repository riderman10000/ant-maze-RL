# AntMaze Deep Reinforcement Learning

A beginner-friendly, research-organized codebase for Gymnasium Robotics AntMaze experiments using Stable-Baselines3.

This is a fresh deep reinforcement learning project. It is not a continuation of a tabular Q-learning GridWorld implementation. AntMaze has continuous actions and dictionary observations, so this code uses Stable-Baselines3 policies and starts with PPO plus `MultiInputPolicy`.

## Why Start Here

`AntMaze_UMazeDense-v5` is the first environment because it is the easiest AntMaze setup to debug:

- U-Maze is the smallest maze layout.
- Dense rewards provide a stronger learning signal than sparse rewards.
- PPO is a simple first baseline before trying off-policy methods.
- Once the pipeline works, the same structure can move to larger or sparse AntMaze variants.

## Project Structure

This repository folder is the project root. You can name the folder `antmaze_drl` if you want it to match the diagram exactly.

```text
.
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- ppo_antmaze_umaze_dense.yaml
|-- src/
|   |-- __init__.py
|   |-- make_env.py
|   |-- train.py
|   |-- evaluate.py
|   |-- visualize.py
|   `-- utils.py
|-- checkpoints/
|-- logs/
`-- results/
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, use a Python version supported by PyTorch, Stable-Baselines3, MuJoCo, and Gymnasium Robotics. Python 3.10-3.12 is a conservative choice for this stack.

Quick import check:

```bash
python -c "import gymnasium, gymnasium_robotics, stable_baselines3, mujoco; print('AntMaze stack OK')"
```

## Train

```bash
python -m src.train --config configs/ppo_antmaze_umaze_dense.yaml
```

Training uses:

- `AntMaze_UMazeDense-v5`
- Stable-Baselines3 `PPO`
- `MultiInputPolicy` for dictionary observations
- TensorBoard logs in `logs/tensorboard`
- Monitor logs in `logs/monitor`
- periodic checkpoints in `checkpoints`
- final model at `checkpoints/ppo_antmaze_umaze_dense_final.zip`

## Evaluate

```bash
python -m src.evaluate \
  --config configs/ppo_antmaze_umaze_dense.yaml \
  --model checkpoints/ppo_antmaze_umaze_dense_final.zip \
  --episodes 10
```

This prints reward, episode length, and success status when the environment reports `success` or `is_success`. It also saves:

```text
results/evaluation_results.csv
results/evaluation_rewards.png
```

## Visualize

Save an MP4 rollout from a trained model:

```bash
python -m src.visualize \
  --config configs/ppo_antmaze_umaze_dense.yaml \
  --model checkpoints/ppo_antmaze_umaze_dense_final.zip
```

Default video output:

```text
results/antmaze_rollout.mp4
```

Run a random policy before training as a sanity check:

```bash
python -m src.visualize --config configs/ppo_antmaze_umaze_dense.yaml --random
```

Use human rendering instead of saving video:

```bash
python -m src.visualize \
  --config configs/ppo_antmaze_umaze_dense.yaml \
  --model checkpoints/ppo_antmaze_umaze_dense_final.zip \
  --render-mode human \
  --no-video
```

Choose a custom video path:

```bash
python -m src.visualize \
  --config configs/ppo_antmaze_umaze_dense.yaml \
  --model checkpoints/ppo_antmaze_umaze_dense_final.zip \
  --video results/my_rollout.mp4
```

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

Then open the local TensorBoard URL printed by the command.

## Configuration

The first experiment lives in:

```text
configs/ppo_antmaze_umaze_dense.yaml
```

Important fields:

```yaml
env_id: AntMaze_UMazeDense-v5
algorithm: PPO
total_timesteps: 100000
learning_rate: 0.0003
n_steps: 2048
batch_size: 64
gamma: 0.99
seed: 42
render_mode: rgb_array
tensorboard_log: logs/tensorboard
monitor_dir: logs/monitor
checkpoint_dir: checkpoints
checkpoint_freq: 25000
result_dir: results
final_model_name: ppo_antmaze_umaze_dense_final
```

## Next Steps

Good follow-up experiments:

- switch to `AntMaze_MediumDense-v5`
- train longer, for example 500k to 2M timesteps
- add SAC as an off-policy baseline
- add TD3 as another continuous-control baseline
- try sparse reward AntMaze variants
- use HER-style replay with an off-policy algorithm if sparse rewards are too hard
- add simple plots from `results/evaluation_results.csv`

## Notes

Gymnasium Robotics environments are registered in `src/make_env.py` before calling `gym.make(...)`. Training intentionally uses Stable-Baselines3 instead of a custom PPO implementation, and no tabular Q-learning code is included.
