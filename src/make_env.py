"""Environment creation for Gymnasium Robotics AntMaze."""

from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

_ROBOTICS_REGISTERED = False


class AntMazeRewardShaping(gym.Wrapper):
    """Add gentle posture and goal-progress shaping for AntMaze training."""

    def __init__(
        self,
        env: gym.Env,
        flip_penalty: float = -0.1,
        progress_reward_scale: float = 1.0,
        vertical_motion_penalty_scale: float = 0.05,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        min_upright_z: float = 0.3,
    ):
        super().__init__(env)
        self.flip_penalty = float(flip_penalty)
        self.progress_reward_scale = float(progress_reward_scale)
        self.vertical_motion_penalty_scale = float(vertical_motion_penalty_scale)
        self.healthy_z_range = healthy_z_range
        self.min_upright_z = float(min_upright_z)
        self.previous_distance: float | None = None
        self.previous_torso_z: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_distance = self._goal_distance(obs)
        self.previous_torso_z = self._torso_z(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_distance = self._goal_distance(obs)
        flipped, torso_z, uprightness = self._is_flipped(obs)

        distance_delta = 0.0
        progress_reward = 0.0
        if self.previous_distance is not None and current_distance is not None:
            distance_delta = self.previous_distance - current_distance
            if not flipped and distance_delta > 0.0:
                progress_reward = self.progress_reward_scale * distance_delta

        posture_penalty = self.flip_penalty if flipped else 0.0
        vertical_motion = 0.0
        if self.previous_torso_z is not None and torso_z is not None:
            vertical_motion = abs(torso_z - self.previous_torso_z)
        vertical_motion_penalty = -self.vertical_motion_penalty_scale * vertical_motion

        shaped_reward = (
            float(reward)
            + progress_reward
            + posture_penalty
            + vertical_motion_penalty
        )

        self.previous_distance = current_distance
        self.previous_torso_z = torso_z
        info["base_reward"] = float(reward)
        info["progress_reward"] = progress_reward
        info["posture_penalty"] = posture_penalty
        info["vertical_motion_penalty"] = vertical_motion_penalty
        info["vertical_motion"] = vertical_motion
        info["distance_delta"] = distance_delta
        info["distance_to_goal"] = current_distance
        info["is_flipped"] = flipped
        info["torso_z"] = torso_z
        info["uprightness"] = uprightness
        return obs, shaped_reward, terminated, truncated, info

    def _goal_distance(self, obs) -> float | None:
        if not isinstance(obs, dict):
            return None
        if "achieved_goal" not in obs or "desired_goal" not in obs:
            return None

        try:
            import numpy as np

            return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        except Exception:
            return None

    def _is_flipped(self, obs) -> tuple[bool, float | None, float | None]:
        torso_z = self._torso_z(obs)
        uprightness = self._uprightness()

        z_unhealthy = False
        if torso_z is not None:
            min_z, max_z = self.healthy_z_range
            z_unhealthy = not (min_z <= torso_z <= max_z)

        tilted = uprightness is not None and uprightness < self.min_upright_z
        return z_unhealthy or tilted, torso_z, uprightness

    def _torso_z(self, obs) -> float | None:
        ant_env = getattr(self.unwrapped, "ant_env", None)
        if ant_env is not None and hasattr(ant_env, "data"):
            try:
                return float(ant_env.data.qpos[2])
            except (TypeError, ValueError, IndexError):
                pass

        if isinstance(obs, dict) and "observation" in obs:
            try:
                return float(obs["observation"][0])
            except (TypeError, ValueError, IndexError):
                pass

        return None

    def _uprightness(self) -> float | None:
        ant_env = getattr(self.unwrapped, "ant_env", None)
        if ant_env is None or not hasattr(ant_env, "data"):
            return None

        try:
            quat = ant_env.data.qpos[3:7]
            w, x, y, z = [float(value) for value in quat]
            norm = (w * w + x * x + y * y + z * z) ** 0.5
            if norm == 0.0:
                return None
            x /= norm
            y /= norm
            return 1.0 - 2.0 * (x * x + y * y)
        except (TypeError, ValueError, IndexError):
            return None


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
    reward_shaping: bool = False,
    flip_penalty: float = -0.1,
    progress_reward_scale: float = 1.0,
    vertical_motion_penalty_scale: float = 0.05,
    healthy_z_range: tuple[float, float] = (0.2, 1.0),
    min_upright_z: float = 0.3,
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

    if reward_shaping:
        env = AntMazeRewardShaping(
            env,
            flip_penalty=flip_penalty,
            progress_reward_scale=progress_reward_scale,
            vertical_motion_penalty_scale=vertical_motion_penalty_scale,
            healthy_z_range=healthy_z_range,
            min_upright_z=min_upright_z,
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
    reward_shaping: bool = False,
    flip_penalty: float = -0.1,
    progress_reward_scale: float = 1.0,
    vertical_motion_penalty_scale: float = 0.05,
    healthy_z_range: tuple[float, float] = (0.2, 1.0),
    min_upright_z: float = 0.3,
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
            reward_shaping=reward_shaping,
            flip_penalty=flip_penalty,
            progress_reward_scale=progress_reward_scale,
            vertical_motion_penalty_scale=vertical_motion_penalty_scale,
            healthy_z_range=healthy_z_range,
            min_upright_z=min_upright_z,
        )

    return _init


def make_vec_env(
    env_id: str,
    n_envs: int = 1,
    render_mode: str | None = None,
    seed: int | None = None,
    monitor_dir: str | Path | None = None,
    vec_env_type: str = "subproc",
    reward_shaping: bool = False,
    flip_penalty: float = -0.1,
    progress_reward_scale: float = 1.0,
    vertical_motion_penalty_scale: float = 0.05,
    healthy_z_range: tuple[float, float] = (0.2, 1.0),
    min_upright_z: float = 0.3,
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
            reward_shaping=reward_shaping,
            flip_penalty=flip_penalty,
            progress_reward_scale=progress_reward_scale,
            vertical_motion_penalty_scale=vertical_motion_penalty_scale,
            healthy_z_range=healthy_z_range,
            min_upright_z=min_upright_z,
        )
        for rank in range(n_envs)
    ]

    if n_envs == 1 or vec_env_type == "dummy":
        return DummyVecEnv(env_fns)
    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns)

    raise ValueError("vec_env_type must be either 'dummy' or 'subproc'")
