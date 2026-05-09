"""Training script for SB3 agents on Gymnasium Robotics AntMaze."""

import argparse
from pathlib import Path

from gymnasium.spaces import Dict as DictSpace
from stable_baselines3.common.callbacks import CheckpointCallback

from src.algorithms import build_model, get_algorithm_name, get_model_class
from src.make_env import make_vec_env
from src.utils import create_dirs, ensure_zip_path, load_config, set_seed


def train(
    config: dict,
    resume_path: str | None = None,
    init_weights_path: str | None = None,
    reset_num_timesteps: bool = False,
) -> None:
    """Train an SB3 agent from a YAML config."""
    if resume_path is not None and init_weights_path is not None:
        raise ValueError("Use either --resume or --init-weights, not both.")

    algorithm = get_algorithm_name(config)

    env_id = config["env_id"]
    total_timesteps = int(config["total_timesteps"])
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    gamma = float(config["gamma"])
    seed = config.get("seed")

    tensorboard_log = Path(config["tensorboard_log"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    monitor_dir = config.get("monitor_dir")
    final_model_name = config["final_model_name"]
    checkpoint_freq = int(config.get("checkpoint_freq", 0))
    progress_bar = bool(config.get("progress_bar", True))
    device = config.get("device", "auto")
    verbose = int(config.get("verbose", 1))
    n_envs = int(config.get("n_envs", 1))
    vec_env_type = str(config.get("vec_env_type", "subproc")).lower()
    reward_shaping = bool(config.get("reward_shaping", False))
    flip_penalty = float(config.get("flip_penalty", -0.1))
    progress_reward_scale = float(config.get("progress_reward_scale", 1.0))
    vertical_motion_penalty_scale = float(config.get("vertical_motion_penalty_scale", 0.05))
    healthy_z_range = tuple(config.get("healthy_z_range", [0.2, 1.0]))
    min_upright_z = float(config.get("min_upright_z", 0.3))

    create_dirs(config)

    if seed is not None:
        seed = int(seed)
        set_seed(seed)

    print("=" * 60)
    print(f"Training {algorithm} on {env_id}")
    print("=" * 60)
    print(f"Version: {config['version']}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Environment copies: {n_envs}")
    print(f"Vector env type: {vec_env_type if n_envs > 1 else 'dummy'}")
    print("Reward source: Gymnasium Robotics environment reward")
    if reward_shaping:
        print("Reward shaping: enabled")
        print(f"  Flip/tilt penalty: {flip_penalty}")
        print(f"  Goal-progress reward scale: {progress_reward_scale}")
        print(f"  Vertical motion penalty scale: {vertical_motion_penalty_scale}")
        print(f"  Healthy z range: {healthy_z_range}")
        print(f"  Minimum uprightness: {min_upright_z}")
    print(f"Device: {device}")
    if resume_path is not None:
        print(f"Resume checkpoint: {resume_path}")
        print(f"Reset timestep counter: {reset_num_timesteps}")
        print(f"{algorithm} hyperparameters will be loaded from the checkpoint.")
    if init_weights_path is not None:
        print(f"Initialize network weights from: {init_weights_path}")
        print("Config hyperparameters and replay buffer settings will be used.")
        print("Replay buffer starts empty after weight initialization.")
    if resume_path is None:
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Gamma: {gamma}")
        if algorithm == "PPO":
            print(f"N steps: {int(config['n_steps'])}")
            print(f"N epochs: {int(config.get('n_epochs', 10))}")
            print(f"GAE lambda: {float(config.get('gae_lambda', 0.95))}")
        else:
            print(f"Buffer size: {int(config.get('buffer_size', 1_000_000))}")
            print(f"Learning starts: {int(config.get('learning_starts', 10_000))}")
            print(f"Train freq: {config.get('train_freq', 1)}")
            print(f"Gradient steps: {int(config.get('gradient_steps', 1))}")
            print(f"Use HER: {bool(config.get('use_her', False))}")
    if seed is not None:
        print(f"Seed: {seed}")
    print("=" * 60)

    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        render_mode=None,
        seed=seed,
        monitor_dir=monitor_dir,
        vec_env_type=vec_env_type,
        reward_shaping=reward_shaping,
        flip_penalty=flip_penalty,
        progress_reward_scale=progress_reward_scale,
        vertical_motion_penalty_scale=vertical_motion_penalty_scale,
        healthy_z_range=healthy_z_range,
        min_upright_z=min_upright_z,
    )
    try:
        if not isinstance(env.observation_space, DictSpace):
            raise TypeError(
                "AntMaze should expose dictionary observations. "
                f"Got observation space: {env.observation_space}"
            )

        if resume_path is not None:
            resume_model_path = ensure_zip_path(resume_path)
            if not resume_model_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_model_path}")

            model_class = get_model_class(algorithm)
            model = model_class.load(
                resume_model_path,
                env=env,
                tensorboard_log=str(tensorboard_log),
                verbose=verbose,
                device=device,
            )
            print(f"Loaded checkpoint with {model.num_timesteps} previous timesteps.")
        else:
            model = build_model(
                config=config,
                env=env,
                tensorboard_log=str(tensorboard_log),
                verbose=verbose,
                seed=seed,
                device=device,
            )
            if init_weights_path is not None:
                init_model_path = ensure_zip_path(init_weights_path)
                if not init_model_path.exists():
                    raise FileNotFoundError(
                        f"Initial weights checkpoint not found: {init_model_path}"
                    )
                model.set_parameters(
                    str(init_model_path),
                    exact_match=False,
                    device=device,
                )
                print(f"Initialized model weights from {init_model_path}.")

        callback = None
        if checkpoint_freq > 0:
            save_freq = max(checkpoint_freq // n_envs, 1)
            callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{final_model_name}_step",
                save_replay_buffer=True,
                save_vecnormalize=False,
            )

        print("\nStarting training...")
        learn_reset_num_timesteps = reset_num_timesteps if resume_path is not None else True
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=final_model_name,
            reset_num_timesteps=learn_reset_num_timesteps,
            progress_bar=progress_bar,
        )

        model_path = ensure_zip_path(checkpoint_dir / final_model_name)
        model.save(model_path)

        print("\nTraining completed!")
        print(f"Final model saved to {model_path}")
        if checkpoint_freq > 0:
            print(f"Periodic checkpoints saved in {checkpoint_dir}")
        print(f"TensorBoard logs saved to {tensorboard_log}")
        print("\nTo view TensorBoard logs, run:")
        print(f"  tensorboard --logdir {tensorboard_log}")
    finally:
        env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train an SB3 agent on AntMaze."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ppo_antmaze_umaze_dense.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to an existing Stable-Baselines3 .zip checkpoint to continue training',
    )
    parser.add_argument(
        '--init-weights',
        type=str,
        default=None,
        help=(
            'Path to a .zip checkpoint used only to initialize network weights. '
            'Unlike --resume, the new config hyperparameters are used.'
        ),
    )
    parser.add_argument(
        '--reset-timesteps',
        action='store_true',
        help='Restart TensorBoard/checkpoint timestep count when resuming',
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(
        config,
        resume_path=args.resume,
        init_weights_path=args.init_weights,
        reset_num_timesteps=args.reset_timesteps,
    )


if __name__ == '__main__':
    main()
