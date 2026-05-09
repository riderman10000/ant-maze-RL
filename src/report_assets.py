"""Generate report-ready figures and tables from AntMaze results."""

import argparse
from pathlib import Path


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    """Parse LABEL=RESULT_DIR or LABEL:RESULT_DIR."""
    if "=" in spec:
        label, path = spec.split("=", maxsplit=1)
    elif ":" in spec:
        label, path = spec.split(":", maxsplit=1)
    else:
        path = spec
        label = Path(spec).name.replace("results-", "")

    label = label.strip()
    if not label:
        raise ValueError(f"Run label is empty in spec: {spec}")
    return label, Path(path.strip())


def _monitor_dir_for_result_dir(result_dir: Path) -> Path:
    """Infer the monitor log directory created by normalize_config()."""
    name = result_dir.name
    if name.startswith("results-"):
        return Path("logs") / f"monitor-{name.removeprefix('results-')}"
    return Path("logs") / f"monitor-{name}"


def _read_monitor_data(monitor_dir: Path):
    """Read SB3 Monitor CSV files from one run."""
    import pandas as pd

    monitor_files = sorted(monitor_dir.glob("**/monitor.csv"))
    frames = []
    for monitor_file in monitor_files:
        df = pd.read_csv(monitor_file, comment="#")
        if {"r", "l"}.issubset(df.columns):
            df["monitor_file"] = str(monitor_file)
            frames.append(df)

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    sort_column = "t" if "t" in data.columns else None
    if sort_column is not None:
        data = data.sort_values(sort_column)
    data["env_steps"] = data["l"].cumsum()
    return data.reset_index(drop=True)


def _estimate_convergence_step(monitor_data, rolling_window: int) -> float | None:
    """Estimate convergence as first step reaching 95% of final rolling return."""
    if monitor_data is None or monitor_data.empty:
        return None

    rolling = monitor_data["r"].rolling(
        window=rolling_window,
        min_periods=1,
    ).mean()
    final_value = float(rolling.iloc[-1])
    if final_value <= 0:
        return None

    threshold = 0.95 * final_value
    reached = monitor_data.loc[rolling >= threshold, "env_steps"]
    if reached.empty:
        return None
    return float(reached.iloc[0])


def _read_evaluation_summary(label: str, result_dir: Path, rolling_window: int) -> dict:
    """Summarize evaluation CSV and monitor logs for one run."""
    import pandas as pd

    eval_path = result_dir / "evaluation_results.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {eval_path}")

    eval_df = pd.read_csv(eval_path)
    success_rate = None
    if "success" in eval_df.columns:
        success_values = eval_df["success"].dropna()
        if not success_values.empty:
            success_bool = success_values.astype(str).str.lower().isin(["true", "1", "1.0"])
            success_rate = float(success_bool.mean())

    monitor_dir = _monitor_dir_for_result_dir(result_dir)
    monitor_data = _read_monitor_data(monitor_dir)
    training_steps = None if monitor_data is None else float(monitor_data["env_steps"].iloc[-1])
    convergence_step = _estimate_convergence_step(monitor_data, rolling_window)
    final_training_return = None
    if monitor_data is not None and not monitor_data.empty:
        final_training_return = float(
            monitor_data["r"].rolling(window=rolling_window, min_periods=1).mean().iloc[-1]
        )

    return {
        "run": label,
        "result_dir": str(result_dir),
        "episodes": int(len(eval_df)),
        "avg_return": float(eval_df["reward"].mean()),
        "std_return": float(eval_df["reward"].std(ddof=0)),
        "success_rate": success_rate,
        "avg_length": float(eval_df["length"].mean()),
        "avg_min_distance": float(eval_df["min_distance"].mean()),
        "avg_final_distance": float(eval_df["final_distance"].mean()),
        "avg_success_step": (
            None
            if "success_step" not in eval_df or eval_df["success_step"].dropna().empty
            else float(eval_df["success_step"].dropna().mean())
        ),
        "training_steps": training_steps,
        "final_training_return_rolling": final_training_return,
        "steps_to_convergence_estimate": convergence_step,
    }


def _save_markdown_table(summary_df, output_path: Path) -> None:
    """Save a Markdown comparison table."""
    display_df = summary_df.copy()
    for column in display_df.columns:
        if display_df[column].dtype.kind in "fc":
            display_df[column] = display_df[column].map(
                lambda value: "" if value != value else f"{value:.3f}"
            )

    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison Markdown table saved to {output_path}")


def _save_table_png(summary_df, output_path: Path) -> None:
    """Save a compact PNG table for report insertion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [
        "run",
        "episodes",
        "avg_return",
        "success_rate",
        "avg_min_distance",
        "avg_final_distance",
        "training_steps",
        "steps_to_convergence_estimate",
    ]
    table_df = summary_df[columns].copy()
    for column in table_df.columns:
        if table_df[column].dtype.kind in "fc":
            table_df[column] = table_df[column].map(
                lambda value: "" if value != value else f"{value:.3f}"
            )

    fig_height = max(2.5, 0.45 * len(table_df) + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Comparison PNG table saved to {output_path}")


def _save_training_curve(
    runs: list[tuple[str, Path]],
    output_path: Path,
    rolling_window: int,
) -> None:
    """Save rolling training return curves from Monitor logs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for label, result_dir in runs:
        monitor_data = _read_monitor_data(_monitor_dir_for_result_dir(result_dir))
        if monitor_data is None or monitor_data.empty:
            print(f"No monitor data found for {label}; skipping training curve.")
            continue

        rolling_return = monitor_data["r"].rolling(
            window=rolling_window,
            min_periods=1,
        ).mean()
        ax.plot(
            monitor_data["env_steps"],
            rolling_return,
            linewidth=2,
            label=label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        print("No training curves were saved because no monitor data was found.")
        return

    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Rolling episode return (window={rolling_window})")
    ax.set_title("AntMaze Training Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Training curve saved to {output_path}")


def generate_report_assets(
    runs: list[tuple[str, Path]],
    output_dir: Path,
    rolling_window: int,
) -> None:
    """Generate comparison table and training curve assets."""
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        _read_evaluation_summary(label, result_dir, rolling_window)
        for label, result_dir in runs
    ]
    summary_df = pd.DataFrame(summaries)

    csv_path = output_dir / "comparison_table.csv"
    markdown_path = output_dir / "comparison_table.md"
    png_path = output_dir / "comparison_table.png"
    training_curve_path = output_dir / "training_curves.png"

    summary_df.to_csv(csv_path, index=False)
    print(f"Comparison CSV saved to {csv_path}")
    _save_markdown_table(summary_df, markdown_path)
    _save_table_png(summary_df, png_path)
    _save_training_curve(runs, training_curve_path, rolling_window)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate report assets from AntMaze evaluation results."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as LABEL=RESULT_DIR. Repeat for each algorithm/run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results-report-assets",
        help="Directory where report-ready assets are written.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Episode window for training curves and convergence estimate.",
    )
    args = parser.parse_args()

    runs = [_parse_run_spec(spec) for spec in args.run]
    generate_report_assets(
        runs=runs,
        output_dir=Path(args.output_dir),
        rolling_window=args.rolling_window,
    )


if __name__ == "__main__":
    main()
