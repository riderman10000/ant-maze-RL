"""Small shared helpers for AntMaze experiments."""

import random
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def create_dirs(config: dict[str, Any]) -> None:
    """Create the standard output directories used by this project."""
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


def save_evaluation_results(
    episodes: list[int],
    rewards: list[float],
    lengths: list[int],
    successes: list[bool | None],
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
