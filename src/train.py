"""Training script for PPO on Gymnasium Robotics AntMaze."""

import argparse
from pathlib import Path

from gymnasium.spaces import Dict as DictSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.make_env import make_env
from src.utils import create_dirs, ensure_zip_path, load_config, set_seed


def train(config: dict) -> None:
    """Train a PPO agent from a YAML config."""
    algorithm = str(config.get("algorithm", "PPO")).upper()
    if algorithm != "PPO":
        raise ValueError("This starter code currently supports only algorithm: PPO")

    env_id = config["env_id"]
    total_timesteps = int(config["total_timesteps"])
    learning_rate = float(config["learning_rate"])
    n_steps = int(config["n_steps"])
    batch_size = int(config["batch_size"])
    gamma = float(config["gamma"])
    gae_lambda = float(config.get("gae_lambda", 0.95))
    n_epochs = int(config.get("n_epochs", 10))
    ent_coef = float(config.get("ent_coef", 0.0))
    seed = config.get("seed")

    tensorboard_log = Path(config["tensorboard_log"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    monitor_dir = config.get("monitor_dir")
    final_model_name = config["final_model_name"]
    checkpoint_freq = int(config.get("checkpoint_freq", 0))
    progress_bar = bool(config.get("progress_bar", True))
    device = config.get("device", "auto")
    verbose = int(config.get("verbose", 1))

    create_dirs(config)

    if seed is not None:
        seed = int(seed)
        set_seed(seed)

    print("=" * 60)
    print(f"Training PPO on {env_id}")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"N steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Gamma: {gamma}")
    print(f"Device: {device}")
    if seed is not None:
        print(f"Seed: {seed}")
    print("=" * 60)

    env = make_env(env_id=env_id, render_mode=None, seed=seed, monitor_dir=monitor_dir)
    try:
        if not isinstance(env.observation_space, DictSpace):
            raise TypeError(
                "AntMaze should expose dictionary observations. "
                f"Got observation space: {env.observation_space}"
            )

        # AntMaze uses dictionary observations, so SB3 needs MultiInputPolicy.
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            tensorboard_log=str(tensorboard_log),
            verbose=verbose,
            seed=seed,
            device=device,
        )

        callback = None
        if checkpoint_freq > 0:
            callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=f"{final_model_name}_step",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )

        print("\nStarting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=final_model_name,
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
        description="Train PPO agent on AntMaze."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ppo_antmaze_umaze_dense.yaml',
        help='Path to config file',
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
