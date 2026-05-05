"""Stable-Baselines3 algorithm helpers."""

from typing import Any

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her import HerReplayBuffer


ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}

OFF_POLICY_ALGORITHMS = {"SAC", "TD3"}


def get_algorithm_name(config: dict[str, Any]) -> str:
    """Return the normalized algorithm name from config."""
    algorithm = str(config.get("algorithm", "PPO")).upper()
    if algorithm not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm `{algorithm}`. Supported: {supported}")
    return algorithm


def get_model_class(config_or_algorithm: dict[str, Any] | str):
    """Return the SB3 model class for a config or algorithm name."""
    if isinstance(config_or_algorithm, dict):
        algorithm = get_algorithm_name(config_or_algorithm)
    else:
        algorithm = str(config_or_algorithm).upper()

    if algorithm not in ALGORITHMS:
        supported = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unsupported algorithm `{algorithm}`. Supported: {supported}")
    return ALGORITHMS[algorithm]


def parse_train_freq(value: Any) -> int | tuple[int, str]:
    """Parse SB3 off-policy train_freq from YAML."""
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), str(value[1])
    raise ValueError("train_freq must be an int or a two-item list like [1, step]")


def _her_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build HER replay buffer kwargs from config."""
    return {
        "n_sampled_goal": int(config.get("her_n_sampled_goal", 4)),
        "goal_selection_strategy": config.get("her_goal_selection_strategy", "future"),
    }


def _action_noise(config: dict[str, Any], env):
    """Create optional NormalActionNoise for TD3."""
    action_noise_std = config.get("action_noise_std")
    if action_noise_std is None:
        return None

    import numpy as np

    action_dim = env.action_space.shape[0]
    return NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=float(action_noise_std) * np.ones(action_dim),
    )


def build_model(
    config: dict[str, Any],
    env,
    tensorboard_log: str,
    verbose: int,
    seed: int | None,
    device: str,
):
    """Create a new SB3 model from config."""
    algorithm = get_algorithm_name(config)
    model_class = get_model_class(algorithm)
    policy = config.get("policy", "MultiInputPolicy")

    common_kwargs = {
        "policy": policy,
        "env": env,
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "gamma": float(config["gamma"]),
        "tensorboard_log": tensorboard_log,
        "verbose": verbose,
        "seed": seed,
        "device": device,
    }

    if algorithm == "PPO":
        return model_class(
            n_steps=int(config["n_steps"]),
            n_epochs=int(config.get("n_epochs", 10)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            ent_coef=float(config.get("ent_coef", 0.0)),
            **common_kwargs,
        )

    off_policy_kwargs = {
        "buffer_size": int(config.get("buffer_size", 1_000_000)),
        "learning_starts": int(config.get("learning_starts", 10_000)),
        "tau": float(config.get("tau", 0.005)),
        "train_freq": parse_train_freq(config.get("train_freq", 1)),
        "gradient_steps": int(config.get("gradient_steps", 1)),
    }

    if bool(config.get("use_her", False)):
        if algorithm not in OFF_POLICY_ALGORITHMS:
            raise ValueError("HER is supported here only with SAC or TD3.")
        off_policy_kwargs["replay_buffer_class"] = HerReplayBuffer
        off_policy_kwargs["replay_buffer_kwargs"] = _her_kwargs(config)

    if algorithm == "SAC":
        return model_class(
            ent_coef=config.get("ent_coef", "auto"),
            target_update_interval=int(config.get("target_update_interval", 1)),
            **off_policy_kwargs,
            **common_kwargs,
        )

    if algorithm == "TD3":
        return model_class(
            action_noise=_action_noise(config, env),
            policy_delay=int(config.get("policy_delay", 2)),
            target_policy_noise=float(config.get("target_policy_noise", 0.2)),
            target_noise_clip=float(config.get("target_noise_clip", 0.5)),
            **off_policy_kwargs,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported algorithm `{algorithm}`")
