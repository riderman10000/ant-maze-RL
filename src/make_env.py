"""Environment creation for Gymnasium Robotics AntMaze."""

from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

_ROBOTICS_REGISTERED = False


def register_robotics_envs() -> None:
    """Register Gymnasium Robotics environments once per Python process."""
    global _ROBOTICS_REGISTERED
    if _ROBOTICS_REGISTERED:
        return

    try:
        import gymnasium_robotics
    except ImportError as exc:
        raise ImportError(
            "gymnasium-robotics is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    # Newer Gymnasium versions use gym.register_envs(...). Older Robotics
    # releases exposed register_robotics_envs(). Support both to keep setup easy.
    if hasattr(gym, "register_envs"):
        gym.register_envs(gymnasium_robotics)
    elif hasattr(gymnasium_robotics, "register_robotics_envs"):
        gymnasium_robotics.register_robotics_envs()

    _ROBOTICS_REGISTERED = True


def make_env(
    env_id: str,
    render_mode: str | None = None,
    seed: int | None = None,
    monitor_dir: str | Path | None = None,
) -> gym.Env:
    """Create a monitored Gymnasium Robotics environment."""
    register_robotics_envs()

    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    try:
        env = gym.make(env_id, **env_kwargs)
    except gym.error.Error as exc:
        raise RuntimeError(
            f"Could not create environment `{env_id}`. Check that "
            "gymnasium-robotics is installed and the environment ID is valid."
        ) from exc

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)

    monitor_file = None
    if monitor_dir is not None:
        monitor_path = Path(monitor_dir)
        monitor_path.mkdir(parents=True, exist_ok=True)
        monitor_file = str(monitor_path / "monitor.csv")

    return Monitor(env, filename=monitor_file)


def _make_env_fn(
    env_id: str,
    rank: int,
    render_mode: str | None = None,
    seed: int | None = None,
    monitor_dir: str | Path | None = None,
) -> Callable[[], gym.Env]:
    """Create a thunk used by Stable-Baselines3 vectorized environments."""

    def _init() -> gym.Env:
        env_seed = None if seed is None else seed + rank
        env_monitor_dir = None
        if monitor_dir is not None:
            env_monitor_dir = Path(monitor_dir) / f"env_{rank}"

        return make_env(
            env_id=env_id,
            render_mode=render_mode,
            seed=env_seed,
            monitor_dir=env_monitor_dir,
        )

    return _init


def make_vec_env(
    env_id: str,
    n_envs: int = 1,
    render_mode: str | None = None,
    seed: int | None = None,
    monitor_dir: str | Path | None = None,
    vec_env_type: str = "subproc",
) -> VecEnv:
    """Create one or more monitored AntMaze environment copies."""
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")

    env_fns = [
        _make_env_fn(
            env_id=env_id,
            rank=rank,
            render_mode=render_mode,
            seed=seed,
            monitor_dir=monitor_dir,
        )
        for rank in range(n_envs)
    ]

    if n_envs == 1 or vec_env_type == "dummy":
        return DummyVecEnv(env_fns)
    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns)

    raise ValueError("vec_env_type must be either 'dummy' or 'subproc'")
