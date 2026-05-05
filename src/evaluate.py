"""Evaluation script for trained AntMaze agents."""

import argparse
from pathlib import Path

from src.algorithms import get_algorithm_name, get_model_class
from src.make_env import make_env
from src.utils import (
    achieved_xy,
    create_dirs,
    ensure_zip_path,
    extract_success,
    goal_xy,
    goal_distance,
    load_config,
    save_distance_plot,
    save_evaluation_plot,
    save_evaluation_results,
    save_rollout_distance_plot,
    save_rollout_trace,
    save_xy_trajectory_plot,
)


def evaluate(
    config: dict,
    model_path: str,
    num_episodes: int = 10,
) -> None:
    """Evaluate a trained SB3 agent."""
    if num_episodes <= 0:
        raise ValueError("--episodes must be greater than zero")

    env_id = config["env_id"]
    algorithm = get_algorithm_name(config)
    result_dir = Path(config["result_dir"])
    seed = config.get("seed")
    if seed is not None:
        seed = int(seed)
    penalize_unhealthy = bool(config.get("penalize_unhealthy", False))
    unhealthy_penalty = float(config.get("unhealthy_penalty", -1.0))
    terminate_on_unhealthy = bool(config.get("terminate_on_unhealthy", False))
    healthy_z_range = tuple(config.get("healthy_z_range", [0.2, 1.0]))

    create_dirs(config)
    model_path = ensure_zip_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("=" * 60)
    print(f"Evaluating {algorithm} model on {env_id}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    env = make_env(
        env_id=env_id,
        render_mode=None,
        seed=seed,
        penalize_unhealthy=penalize_unhealthy,
        unhealthy_penalty=unhealthy_penalty,
        terminate_on_unhealthy=terminate_on_unhealthy,
        healthy_z_range=healthy_z_range,
    )
    model_class = get_model_class(algorithm)
    model = model_class.load(model_path, env=env)

    episodes = []
    rewards = []
    lengths = []
    successes = []
    initial_distances = []
    min_distances = []
    final_distances = []
    success_steps = []
    unhealthy_terminations = []
    unhealthy_steps = []
    unhealthy_penalties = []
    best_rollout = None
    best_min_distance = None

    try:
        for episode in range(num_episodes):
            episode_seed = None if seed is None else seed + episode
            obs, _ = env.reset(seed=episode_seed)
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_success = None
            min_distance = goal_distance(obs)
            initial_distance = min_distance
            final_distance = min_distance
            first_success_step = None
            episode_unhealthy_termination = False
            episode_unhealthy_steps = 0
            episode_unhealthy_penalty = 0.0
            distance_history = []
            xy_history = []
            step_history = []
            rollout_goal = goal_xy(obs)

            initial_xy = achieved_xy(obs)
            if initial_distance is not None:
                distance_history.append(initial_distance)
                step_history.append(0)
            if initial_xy is not None:
                xy_history.append(initial_xy)

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                done = terminated or truncated
                if info.get("unhealthy", False):
                    episode_unhealthy_steps += 1
                    episode_unhealthy_penalty += float(info.get("unhealthy_penalty", 0.0))
                if info.get("unhealthy_termination", False):
                    episode_unhealthy_termination = True

                distance = goal_distance(obs)
                if distance is not None:
                    final_distance = distance
                    distance_history.append(distance)
                    step_history.append(episode_length)
                    if min_distance is None:
                        min_distance = distance
                    else:
                        min_distance = min(min_distance, distance)

                xy = achieved_xy(obs)
                if xy is not None:
                    xy_history.append(xy)

                success = extract_success(info)
                if success is not None:
                    if success and first_success_step is None:
                        first_success_step = episode_length
                    episode_success = success if episode_success is None else episode_success or success

            final_distance = goal_distance(obs)
            if final_distance is not None and not distance_history:
                distance_history.append(final_distance)
                step_history.append(episode_length)

            episodes.append(episode + 1)
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(episode_success)
            initial_distances.append(initial_distance)
            min_distances.append(min_distance)
            final_distances.append(final_distance)
            success_steps.append(first_success_step)
            unhealthy_terminations.append(episode_unhealthy_termination)
            unhealthy_steps.append(episode_unhealthy_steps)
            unhealthy_penalties.append(episode_unhealthy_penalty)

            if min_distance is not None:
                if best_min_distance is None or min_distance < best_min_distance:
                    best_min_distance = min_distance
                    best_rollout = {
                        "episode": episode + 1,
                        "steps": step_history,
                        "distances": distance_history,
                        "xy_positions": xy_history,
                        "goal": rollout_goal,
                    }

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
                f"Final dist: {final_distance if final_distance is not None else float('nan'):.2f} | "
                f"Unhealthy steps: {episode_unhealthy_steps}"
            )
    finally:
        env.close()

    success_values = [success for success in successes if success is not None]
    print("=" * 60)
    print("Evaluation Summary:")
    print(f"  Average reward: {sum(rewards) / len(rewards):.2f}")
    print(f"  Average length: {sum(lengths) / len(lengths):.2f}")
    valid_min_distances = [distance for distance in min_distances if distance is not None]
    valid_final_distances = [distance for distance in final_distances if distance is not None]
    if valid_min_distances:
        print(f"  Average minimum distance: {sum(valid_min_distances) / len(valid_min_distances):.2f}")
    if valid_final_distances:
        print(f"  Average final distance: {sum(valid_final_distances) / len(valid_final_distances):.2f}")
    if penalize_unhealthy:
        print(f"  Average unhealthy steps: {sum(unhealthy_steps) / len(unhealthy_steps):.2f}")
        print(f"  Average unhealthy penalty: {sum(unhealthy_penalties) / len(unhealthy_penalties):.2f}")
    if terminate_on_unhealthy:
        fall_rate = sum(unhealthy_terminations) / len(unhealthy_terminations) * 100
        print(f"  Unhealthy termination rate: {fall_rate:.1f}%")
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
    save_evaluation_results(
        episodes,
        rewards,
        lengths,
        successes,
        initial_distances,
        min_distances,
        final_distances,
        success_steps,
        unhealthy_terminations,
        unhealthy_steps,
        unhealthy_penalties,
        csv_path,
    )

    plot_path = result_dir / "evaluation_rewards.png"
    save_evaluation_plot(episodes, rewards, successes, plot_path)

    distance_plot_path = result_dir / "evaluation_distances.png"
    save_distance_plot(
        episodes,
        min_distances,
        final_distances,
        successes,
        distance_plot_path,
    )

    if best_rollout is not None:
        best_episode = best_rollout["episode"]
        rollout_trace_path = result_dir / "best_rollout_trace.csv"
        save_rollout_trace(
            best_rollout["steps"],
            best_rollout["distances"],
            best_rollout["xy_positions"],
            best_rollout["goal"],
            rollout_trace_path,
        )

        rollout_distance_path = result_dir / "best_rollout_distance_over_time.png"
        save_rollout_distance_plot(
            best_rollout["steps"],
            best_rollout["distances"],
            rollout_distance_path,
            title=f"Best Rollout Distance Over Time (Episode {best_episode})",
        )

        rollout_xy_path = result_dir / "best_rollout_xy_trajectory.png"
        save_xy_trajectory_plot(
            best_rollout["xy_positions"],
            best_rollout["goal"],
            rollout_xy_path,
            title=f"Best Rollout XY Trajectory (Episode {best_episode})",
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SB3 agent on AntMaze."
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
