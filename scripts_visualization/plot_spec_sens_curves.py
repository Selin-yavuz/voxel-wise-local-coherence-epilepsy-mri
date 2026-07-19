from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sort_key(row: pd.Series):
    return (
        str(row["sequence"]),
        int(row["primary_rate"]),
        int(row["secondary_rate"]),
        int(row["power"]),
        str(row["shape"]),
        int(row["step"]),
    )


def build_label(row: pd.Series) -> str:
    return (
        f"{row['sequence']} | "
        f"PR{int(row['primary_rate'])} SR{int(row['secondary_rate'])} "
        f"| PW{int(row['power'])} | {row['shape']} | step{int(row['step'])}"
    )


def marker_for_shape(shape: str) -> str:
    if shape == "circle":
        return "o"
    if shape == "square":
        return "s"
    return "o"


def linestyle_for_shape(shape: str) -> str:
    if shape == "circle":
        return "--"
    if shape == "square":
        return "-"
    return "-"


def marker_size_for_step(step: int, min_size: float = 5.0, max_size: float = 12.0,
                         step_min: int = 1, step_max: int = 10) -> float:
    if step_max <= step_min:
        return max_size
    frac = (int(step) - step_min) / (step_max - step_min)
    return min_size + frac * (max_size - min_size)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot sensitivity vs specificity threshold curves for selected models."
    )

    ap.add_argument(
        "--input",
        type=str,
        default="outputs/tables/cv_single_selected_curve_summary.csv",
        help="Input CSV from selected curve summary.",
    )

    ap.add_argument(
        "--output",
        type=str,
        default="outputs/figures/sens_vs_spec_curves.png",
        help="Output figure path.",
    )

    ap.add_argument(
        "--title",
        type=str,
        default="Sensitivity vs specificity threshold",
        help="Figure title.",
    )

    ap.add_argument(
        "--linewidth",
        type=float,
        default=1.8,
        help="Constant line width for all curves.",
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Constant transparency for all curves.",
    )

    ap.add_argument(
        "--marker-size-min",
        type=float,
        default=5.0,
        help="Minimum marker size.",
    )

    ap.add_argument(
        "--marker-size-max",
        type=float,
        default=12.0,
        help="Maximum marker size.",
    )

    ap.add_argument(
        "--legend-fontsize",
        type=float,
        default=8,
        help="Legend font size.",
    )

    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved figure DPI.",
    )

    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "spec_min",
        "model_id",
        "sequence",
        "shape",
        "primary_rate",
        "secondary_rate",
        "step",
        "power",
        "sens_mean",
    ]
    validate_columns(df, required_cols)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    # build per-model metadata
    model_meta = (
        df.sort_values(["model_id", "spec_min"])
        .groupby("model_id", as_index=False)
        .first()
        .copy()
    )

    model_meta["sort_key"] = model_meta.apply(sort_key, axis=1)
    model_meta["label"] = model_meta.apply(build_label, axis=1)
    model_meta = model_meta.sort_values("sort_key").reset_index(drop=True)

    ordered_models = model_meta["model_id"].tolist()
    label_map = dict(zip(model_meta["model_id"], model_meta["label"]))

    step_min = int(model_meta["step"].min())
    step_max = int(model_meta["step"].max())

    plt.figure(figsize=(11, 7))

    # plot smaller steps first, larger steps later so larger markers remain visible
    ordered_models = sorted(
        ordered_models,
        key=lambda mid: (
            int(model_meta.loc[model_meta["model_id"] == mid, "step"].iloc[0]),
            label_map[mid],
        )
    )

    for model_id in ordered_models:
        sub = df[df["model_id"] == model_id].sort_values("spec_min")
        if sub.empty:
            continue

        shape = str(sub.iloc[0]["shape"])
        step = int(sub.iloc[0]["step"])

        marker_size = marker_size_for_step(
            step=step,
            min_size=args.marker_size_min,
            max_size=args.marker_size_max,
            step_min=step_min,
            step_max=step_max,
        )

        plt.plot(
            sub["spec_min"],
            sub["sens_mean"],
            linestyle=linestyle_for_shape(shape),
            marker=marker_for_shape(shape),
            linewidth=args.linewidth,
            alpha=args.alpha,
            markersize=marker_size,
            markerfacecolor="none",
            markeredgewidth=1.4,
            zorder=step,
            label=label_map[model_id],
        )

    plt.xlabel("Required training specificity threshold")
    plt.ylabel("Mean test sensitivity")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)

    # stricter specificity on left
    #plt.gca().invert_xaxis()

    plt.legend(fontsize=args.legend_fontsize, loc="best")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()