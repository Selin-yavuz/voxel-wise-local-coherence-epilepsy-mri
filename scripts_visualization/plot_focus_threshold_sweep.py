from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_step_label(step: int) -> str:
    return f"Step {int(step)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot focused threshold sweep: sensitivity and specificity vs threshold."
    )

    ap.add_argument(
        "--input",
        type=str,
        default="outputs/tables/focus_t1_tra_circle_PR3_SR7_PW1_summary.csv",
        help="Input summary CSV from focused threshold sweep.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="outputs/figures/focus_threshold_sweep.png",
        help="Output figure path.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Sensitivity and specificity across thresholds",
        help="Figure title.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI.",
    )
    ap.add_argument(
        "--fig-width",
        type=float,
        default=11.0,
        help="Figure width in inches.",
    )
    ap.add_argument(
        "--fig-height",
        type=float,
        default=7.0,
        help="Figure height in inches.",
    )
    ap.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
        help="Line width.",
    )
    ap.add_argument(
        "--marker-size",
        type=float,
        default=5.0,
        help="Marker size.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Line alpha.",
    )
    ap.add_argument(
        "--legend-fontsize",
        type=float,
        default=9.0,
        help="Legend font size.",
    )
    ap.add_argument(
        "--show-train",
        action="store_true",
        help="Also plot train sensitivity/specificity with lower alpha.",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "step",
        "threshold",
        "threshold_index",
        "sens_test_mean",
        "spec_test_mean",
    ]
    validate_columns(df, required_cols)

    if args.show_train:
        validate_columns(
            df,
            ["sens_train_mean", "spec_train_mean"],
        )

    if df.empty:
        raise ValueError("Input CSV is empty.")

    df = df.sort_values(["step", "threshold_index"]).copy()

    steps = sorted(df["step"].astype(int).unique().tolist())
    if not steps:
        raise ValueError("No steps found in input CSV.")

    plt.figure(figsize=(args.fig_width, args.fig_height))

    cmap = plt.get_cmap("tab10")

    legend_handles = []
    legend_labels = []

    for i, step in enumerate(steps):
        sub = df[df["step"].astype(int) == int(step)].sort_values("threshold_index")
        if sub.empty:
            continue

        color = cmap(i % 10)
        step_label = build_step_label(step)

        line_spec, = plt.plot(
            sub["threshold"],
            sub["spec_test_mean"],
            linestyle="-",
            marker="o",
            linewidth=args.linewidth,
            markersize=args.marker_size,
            alpha=args.alpha,
            color=color,
            label=f"{step_label} specificity",
        )

        line_sens, = plt.plot(
            sub["threshold"],
            sub["sens_test_mean"],
            linestyle=":",
            marker="o",
            linewidth=args.linewidth,
            markersize=args.marker_size,
            alpha=args.alpha,
            color=color,
            label=f"{step_label} sensitivity",
        )

        legend_handles.extend([line_spec, line_sens])
        legend_labels.extend(
            [f"{step_label} specificity", f"{step_label} sensitivity"]
        )

        if args.show_train:
            plt.plot(
                sub["threshold"],
                sub["spec_train_mean"],
                linestyle="-",
                linewidth=max(args.linewidth - 0.5, 1.0),
                alpha=0.35,
                color=color,
            )
            plt.plot(
                sub["threshold"],
                sub["sens_train_mean"],
                linestyle=":",
                linewidth=max(args.linewidth - 0.5, 1.0),
                alpha=0.35,
                color=color,
            )

    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title(args.title)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)

    plt.legend(
        legend_handles,
        legend_labels,
        fontsize=args.legend_fontsize,
        loc="best",
        ncol=3,
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()