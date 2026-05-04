"""Visualize trained or random AntMaze rollouts."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import PPO

from src.make_env import make_env
from src.utils import create_dirs, ensure_zip_path, extract_success, goal_distance, load_config


def _default_video_name(random_policy: bool) -> str:
    if random_policy:
        return "antmaze_random_rollout.mp4"
    return "antmaze_rollout.mp4"


def _resolve_video_path(
    config: dict,
    video_arg: str | None,
    random_policy: bool,
) -> Path:
    """Resolve --video as either an MP4 path or an output directory."""
    video_name = _default_video_name(random_policy)
    if video_arg is None:
        return Path(config["result_dir"]) / video_name

    path = Path(video_arg)
    if path.suffix.lower() == ".mp4":
        return path

    if str(video_arg).endswith(("/", "\\")) or path.is_dir():
        return path / video_name

    return path.with_suffix(".mp4")


def _save_video(frames: list, output_path: Path, fps: int) -> None:
    """Write collected RGB frames to an MP4 file."""
    if not frames:
        print("No rgb_array frames were captured; video was not saved.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved to {output_path}")


def visualize(
    config: dict,
    model_path: str | None = None,
    num_episodes: int = 1,
    start_episode: int = 1,
    random_policy: bool = False,
    render_mode: str | None = None,
    video_path: str | None = None,
    save_video: bool = True,
) -> None:
    """Run rollouts with either a trained model or random actions."""
    if num_episodes <= 0:
        raise ValueError("--episodes must be greater than zero")
    if start_episode <= 0:
        raise ValueError("--start-episode must be greater than zero")

    if model_path is None:
        random_policy = True

    env_id = config["env_id"]
    seed = config.get("seed")
    if seed is not None:
        seed = int(seed)

    if render_mode is None:
        render_mode = config.get("render_mode", "rgb_array")
    env_render_mode = None if render_mode == "none" else render_mode

    output_video = None
    if save_video and env_render_mode == "rgb_array":
        output_video = _resolve_video_path(config, video_path, random_policy)
    elif video_path is not None:
        raise ValueError("Video saving requires --render-mode rgb_array")

    create_dirs(config)

    policy_label = "random policy" if random_policy else "trained model"
    print("=" * 60)
    print(f"Visualizing {policy_label} on {env_id}")
    if not random_policy:
        print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Starting evaluation episode: {start_episode}")
    print(f"Render mode: {env_render_mode}")
    if output_video is not None:
        print(f"Video output: {output_video}")
    print("=" * 60)

    env = make_env(env_id=env_id, render_mode=env_render_mode, seed=seed)
    model = None
    if not random_policy:
        model_file = ensure_zip_path(model_path)
        if not model_file.exists():
            env.close()
            raise FileNotFoundError(f"Model file not found: {model_file}")
        model = PPO.load(model_file, env=env)

    frames = []
    fps = int(getattr(env, "metadata", {}).get("render_fps", 30))

    try:
        for episode_offset in range(num_episodes):
            eval_episode = start_episode + episode_offset
            episode_seed = None if seed is None else seed + eval_episode - 1
            obs, _ = env.reset(seed=episode_seed)
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_success = None
            min_distance = goal_distance(obs)

            if env_render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            while not done:
                if random_policy:
                    action = env.action_space.sample()
                else:
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

                if env_render_mode == "rgb_array":
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                elif env_render_mode == "human":
                    env.render()

            if episode_success is None:
                status = "UNKNOWN"
            else:
                status = "SUCCESS" if episode_success else "FAIL"

            final_distance = goal_distance(obs)
            print(
                f"Episode {eval_episode:3d} | "
                f"Reward: {episode_reward:10.2f} | "
                f"Length: {episode_length:5d} | "
                f"Status: {status} | "
                f"Min dist: {min_distance if min_distance is not None else float('nan'):.2f} | "
                f"Final dist: {final_distance if final_distance is not None else float('nan'):.2f}"
            )
    finally:
        env.close()

    if output_video is not None:
        _save_video(frames, output_video, fps=fps)

    print("=" * 60)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize AntMaze behavior with a trained or random policy."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_antmaze_umaze_dense.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (.zip file). Omit with --random.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=1,
        help="Evaluation episode number to start replaying from",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions as a pre-training sanity check",
    )
    parser.add_argument(
        "--render-mode",
        choices=["rgb_array", "human", "none"],
        default=None,
        help="Override the render mode from the config",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="MP4 output path or directory. Defaults to results/antmaze_rollout.mp4.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Run the rollout without saving an rgb_array video",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    visualize(
        config=config,
        model_path=args.model,
        num_episodes=args.episodes,
        start_episode=args.start_episode,
        random_policy=args.random,
        render_mode=args.render_mode,
        video_path=args.video,
        save_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
