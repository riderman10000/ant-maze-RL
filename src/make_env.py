"""Environment creation for Gymnasium Robotics AntMaze."""

from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

_ROBOTICS_REGISTERED = False


class UnhealthyAntPenalty(gym.Wrapper):
    """Apply a per-step penalty when the Ant falls or flips."""

    def __init__(
        self,
        env: gym.Env,
        penalize_unhealthy: bool = True,
        unhealthy_penalty: float = -1.0,
        terminate_on_unhealthy: bool = False,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
    ):
        super().__init__(env)
        self.penalize_unhealthy = penalize_unhealthy
        self.unhealthy_penalty = unhealthy_penalty
        self.terminate_on_unhealthy = terminate_on_unhealthy
        self.healthy_z_range = healthy_z_range

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        unhealthy = self._is_unhealthy(obs)
        penalty = self.unhealthy_penalty if unhealthy and self.penalize_unhealthy else 0.0
        reward = float(reward) + penalty

        if unhealthy and self.terminate_on_unhealthy and not terminated:
            terminated = True

        info["unhealthy"] = unhealthy
        info["unhealthy_penalty"] = penalty
        info["unhealthy_termination"] = unhealthy and terminated
        return obs, reward, terminated, truncated, info

    def _is_unhealthy(self, obs) -> bool:
        if isinstance(obs, dict) and "observation" in obs:
            try:
                torso_z = float(obs["observation"][0])
                min_z, max_z = self.healthy_z_range
                return not (min_z <= torso_z <= max_z)
            except (TypeError, ValueError, IndexError):
                pass

        ant_env = getattr(self.unwrapped, "ant_env", None)
        if ant_env is not None and hasattr(ant_env, "is_healthy"):
            return not bool(ant_env.is_healthy)

        return False


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
    penalize_unhealthy: bool = False,
    unhealthy_penalty: float = -1.0,
    terminate_on_unhealthy: bool = False,
    healthy_z_range: tuple[float, float] = (0.2, 1.0),
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

    if penalize_unhealthy or terminate_on_unhealthy:
        env = UnhealthyAntPenalty(
            env,
            penalize_unhealthy=penalize_unhealthy,
            unhealthy_penalty=unhealthy_penalty,
            terminate_on_unhealthy=terminate_on_unhealthy,
            healthy_z_range=healthy_z_range,
        )

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
    penalize_unhealthy: bool = False,
    unhealthy_penalty: float = -1.0,
    terminate_on_unhealthy: bool = False,
    healthy_z_range: tuple[float, float] = (0.2, 1.0),
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
            penalize_unhealthy=penalize_unhealthy,
            unhealthy_penalty=unhealthy_penalty,
            terminate_on_unhealthy=terminate_on_unhealthy,
            healthy_z_range=healthy_z_range,
        )

    return _init


def make_vec_env(
    env_id: str,
    n_envs: int = 1,
    render_mode: str | None = None,
    seed: int | None = None,
    monitor_dir: str | Path | None = None,
    vec_env_type: str = "subproc",
    penalize_unhealthy: bool = False,
    unhealthy_penalty: float = -1.0,
    terminate_on_unhealthy: bool = False,
    healthy_z_range: tuple[float, float] = (0.2, 1.0),
) -> VecEnv:
    """Create multiple monitored AntMaze environment copies for one PPO policy."""
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")

    env_fns = [
        _make_env_fn(
            env_id=env_id,
            rank=rank,
            render_mode=render_mode,
            seed=seed,
            monitor_dir=monitor_dir,
            penalize_unhealthy=penalize_unhealthy,
            unhealthy_penalty=unhealthy_penalty,
            terminate_on_unhealthy=terminate_on_unhealthy,
            healthy_z_range=healthy_z_range,
        )
        for rank in range(n_envs)
    ]

    if n_envs == 1 or vec_env_type == "dummy":
        return DummyVecEnv(env_fns)
    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns)

    raise ValueError("vec_env_type must be either 'dummy' or 'subproc'")
