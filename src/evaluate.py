"""Evaluation script for trained AntMaze agents."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.make_env import make_env
from src.utils import (
    create_dirs,
    ensure_zip_path,
    extract_success,
    goal_distance,
    load_config,
    save_evaluation_plot,
    save_evaluation_results,
)


def evaluate(
    config: dict,
    model_path: str,
    num_episodes: int = 10,
) -> None:
    """Evaluate a trained PPO agent."""
    if num_episodes <= 0:
        raise ValueError("--episodes must be greater than zero")

    env_id = config["env_id"]
    result_dir = Path(config["result_dir"])
    seed = config.get("seed")
    if seed is not None:
        seed = int(seed)

    create_dirs(config)
    model_path = ensure_zip_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("=" * 60)
    print(f"Evaluating model on {env_id}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    env = make_env(env_id=env_id, render_mode=None, seed=seed)
    model = PPO.load(model_path, env=env)

    episodes = []
    rewards = []
    lengths = []
    successes = []

    try:
        for episode in range(num_episodes):
            episode_seed = None if seed is None else seed + episode
            obs, _ = env.reset(seed=episode_seed)
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_success = None
            min_distance = goal_distance(obs)

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                done = terminated or truncated

                distance = goal_distance(obs)
                if distance is not None:
                    if min_distance is None:
                        min_distance = distance
                    else:
                        min_distance = min(min_distance, distance)

                success = extract_success(info)
                if success is not None:
                    episode_success = success if episode_success is None else episode_success or success

            final_distance = goal_distance(obs)

            episodes.append(episode + 1)
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(episode_success)

            if episode_success is None:
                status = "UNKNOWN"
            else:
                status = "SUCCESS" if episode_success else "FAIL"

            print(
                f"Episode {episode + 1:3d} | "
                f"Reward: {episode_reward:10.2f} | "
                f"Length: {episode_length:5d} | "
                f"Status: {status} | "
                f"Min dist: {min_distance if min_distance is not None else float('nan'):.2f} | "
                f"Final dist: {final_distance if final_distance is not None else float('nan'):.2f}"
            )
    finally:
        env.close()

    success_values = [success for success in successes if success is not None]
    print("=" * 60)
    print("Evaluation Summary:")
    print(f"  Average reward: {sum(rewards) / len(rewards):.2f}")
    print(f"  Average length: {sum(lengths) / len(lengths):.2f}")
    if len(success_values) > 0:
        success_rate = sum(success_values) / len(success_values) * 100
        print(
            f"  Success rate: {success_rate:.1f}% "
            f"({len(success_values)}/{len(successes)} episodes reported success)"
        )
    else:
        print("  Success rate: not reported by this environment")
    print("=" * 60)

    csv_path = result_dir / "evaluation_results.csv"
    save_evaluation_results(episodes, rewards, lengths, successes, csv_path)

    plot_path = result_dir / "evaluation_rewards.png"
    save_evaluation_plot(episodes, rewards, successes, plot_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on AntMaze."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ppo_antmaze_umaze_dense.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.zip file)',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes',
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Evaluate
    evaluate(config, args.model, args.episodes)


if __name__ == '__main__':
    main()
