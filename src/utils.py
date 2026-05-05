"""Small shared helpers for AntMaze experiments."""

import random
from pathlib import Path
from typing import Any

import yaml


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill config fields that are derived from algorithm/version."""
    algorithm = str(config.get("algorithm", "PPO")).lower()
    version = str(config.get("version", config.get("experiment_version", "v1")))
    config["version"] = version

    run_id = f"{algorithm}-{version}"
    run_name = run_id.replace("-", "_")

    if not config.get("result_dir") or str(config.get("result_dir")).lower() == "auto":
        config["result_dir"] = f"results-{run_id}"
    if not config.get("checkpoint_dir") or str(config.get("checkpoint_dir")).lower() == "auto":
        config["checkpoint_dir"] = f"checkpoints-{run_id}"
    if not config.get("tensorboard_log") or str(config.get("tensorboard_log")).lower() == "auto":
        config["tensorboard_log"] = f"logs/tensorboard-{run_id}"
    if not config.get("monitor_dir") or str(config.get("monitor_dir")).lower() == "auto":
        config["monitor_dir"] = f"logs/monitor-{run_id}"
    if not config.get("final_model_name") or str(config.get("final_model_name")).lower() == "auto":
        config["final_model_name"] = f"{run_name}_final"

    return config


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return normalize_config(config)


def create_dirs(config: dict[str, Any]) -> None:
    """Create the standard output directories used by this project."""
    normalize_config(config)
    dirs = [
        config.get("tensorboard_log"),
        config.get("monitor_dir"),
        config.get("checkpoint_dir"),
        config.get("result_dir"),
    ]
    for dir_path in dirs:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible runs."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_zip_path(path: str | Path) -> Path:
    """Return a model path with SB3's expected .zip suffix."""
    model_path = Path(path)
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    return model_path


def extract_success(info: dict[str, Any]) -> bool | None:
    """Read AntMaze success flags from an info dictionary when present."""
    for key in ("is_success", "success"):
        if key in info:
            value = info[key]
            if hasattr(value, "item"):
                value = value.item()
            return bool(value)
    return None


def goal_distance(obs: dict[str, Any]) -> float | None:
    """Return distance between achieved_goal and desired_goal when available."""
    if "achieved_goal" not in obs or "desired_goal" not in obs:
        return None

    try:
        import numpy as np

        return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
    except Exception:
        return None


def goal_xy(obs: dict[str, Any]) -> tuple[float, float] | None:
    """Return the desired goal xy position when available."""
    if "desired_goal" not in obs:
        return None

    try:
        goal = obs["desired_goal"]
        return float(goal[0]), float(goal[1])
    except Exception:
        return None


def achieved_xy(obs: dict[str, Any]) -> tuple[float, float] | None:
    """Return the ant achieved_goal xy position when available."""
    if "achieved_goal" not in obs:
        return None

    try:
        achieved = obs["achieved_goal"]
        return float(achieved[0]), float(achieved[1])
    except Exception:
        return None


def save_evaluation_results(
    episodes: list[int],
    rewards: list[float],
    lengths: list[int],
    successes: list[bool | None],
    initial_distances: list[float | None] | None,
    min_distances: list[float | None] | None,
    final_distances: list[float | None] | None,
    success_steps: list[int | None] | None,
    unhealthy_terminations: list[bool] | None,
    unhealthy_steps: list[int] | None,
    unhealthy_penalties: list[float] | None,
    output_path: str | Path,
) -> Path:
    """Save evaluation results to CSV and return the output path."""
    import pandas as pd

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "episode": episodes,
            "reward": rewards,
            "length": lengths,
            "success": successes,
            "initial_distance": initial_distances,
            "min_distance": min_distances,
            "final_distance": final_distances,
            "success_step": success_steps,
            "unhealthy_termination": unhealthy_terminations,
            "unhealthy_steps": unhealthy_steps,
            "unhealthy_penalty": unhealthy_penalties,
        }
    )
    df.to_csv(path, index=False)
    print(f"Evaluation results saved to {path}")
    return path


def save_evaluation_plot(
    episodes: list[int],
    rewards: list[float],
    successes: list[bool | None],
    output_path: str | Path,
) -> Path:
    """Save a simple reward-per-episode plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, rewards, marker="o", label="Episode reward")

    reported_successes = [success for success in successes if success is not None]
    if len(reported_successes) > 0:
        colors = [
            "tab:gray" if success is None else "tab:green" if success else "tab:red"
            for success in successes
        ]
        ax.scatter(episodes, rewards, c=colors, zorder=3, label="Success flag")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("AntMaze Evaluation Rewards")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Evaluation plot saved to {path}")
    return path


def save_distance_plot(
    episodes: list[int],
    min_distances: list[float | None],
    final_distances: list[float | None],
    successes: list[bool | None],
    output_path: str | Path,
) -> Path:
    """Save final/min distance-to-goal curves across evaluation episodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, final_distances, marker="o", label="Final distance")
    ax.plot(episodes, min_distances, marker="s", label="Minimum distance")
    ax.axhline(0.45, color="tab:green", linestyle="--", linewidth=1, label="Success radius")

    reported_successes = [success for success in successes if success is not None]
    if len(reported_successes) > 0:
        success_x = [episode for episode, success in zip(episodes, successes) if success]
        success_y = [
            final_distance
            for final_distance, success in zip(final_distances, successes)
            if success
        ]
        if success_x:
            ax.scatter(success_x, success_y, color="tab:green", zorder=3, label="Success")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Distance to goal")
    ax.set_title("AntMaze Distance Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Distance plot saved to {path}")
    return path


def save_rollout_trace(
    steps: list[int],
    distances: list[float],
    xy_positions: list[tuple[float, float]],
    goal: tuple[float, float] | None,
    output_path: str | Path,
) -> Path:
    """Save per-step distance and xy position for one rollout."""
    import pandas as pd

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    goal_x = None if goal is None else goal[0]
    goal_y = None if goal is None else goal[1]
    df = pd.DataFrame(
        {
            "step": steps,
            "distance_to_goal": distances,
            "x": [xy[0] for xy in xy_positions],
            "y": [xy[1] for xy in xy_positions],
            "goal_x": goal_x,
            "goal_y": goal_y,
        }
    )
    df.to_csv(path, index=False)
    print(f"Rollout trace saved to {path}")
    return path


def save_rollout_distance_plot(
    steps: list[int],
    distances: list[float],
    output_path: str | Path,
    title: str = "Distance to Goal Over Time",
) -> Path:
    """Save distance-to-goal over time for one rollout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, distances)
    ax.axhline(0.45, color="tab:green", linestyle="--", linewidth=1, label="Success radius")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Rollout distance plot saved to {path}")
    return path


def save_xy_trajectory_plot(
    xy_positions: list[tuple[float, float]],
    goal: tuple[float, float] | None,
    output_path: str | Path,
    title: str = "Ant XY Trajectory",
) -> Path:
    """Save the ant xy trajectory and goal location for one rollout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs = [xy[0] for xy in xy_positions]
    ys = [xy[1] for xy in xy_positions]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, linewidth=2, label="Ant trajectory")
    if xs and ys:
        ax.scatter(xs[0], ys[0], color="tab:blue", marker="o", zorder=3, label="Start")
        ax.scatter(xs[-1], ys[-1], color="tab:orange", marker="x", zorder=3, label="End")
    if goal is not None:
        ax.scatter(goal[0], goal[1], color="tab:green", marker="*", s=180, zorder=4, label="Goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"XY trajectory plot saved to {path}")
    return path
