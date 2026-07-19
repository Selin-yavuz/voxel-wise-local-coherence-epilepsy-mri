

from __future__ import annotations

import argparse
import pandas as pd

from vic.paths import VICPaths
from vic.analysis.cv_stat import run_cv_on_scores, summarize_cv_results


def summarize_selected_curve_results(
    df_folds: pd.DataFrame,
    *,
    min_folds_passed: int,
) -> pd.DataFrame:
    """
    Summarize fold-level results for selected models across multiple specificity thresholds.
    Keeps per-model, per-spec_min summaries.
    """
    pass_mask = (df_folds["passed_auc"]) & (df_folds["passed_threshold"])

    df_summary = (
        df_folds[pass_mask]
        .groupby(
            ["spec_min", "sequence", "shape", "primary_rate", "secondary_rate", "step", "power", "model_id"]
        )
        .agg(
            n_folds_passed=("fold", "nunique"),
            auc_train_mean=("auc_train", "mean"),
            auc_train_max=("auc_train", "max"),
            thr_star_min=("thr_star", "min"),
            thr_star_max=("thr_star", "max"),
            thr_star_mean=("thr_star", "mean"),
            sens_mean=("sens_test", "mean"),
            sens_std=("sens_test", "std"),
            spec_mean=("spec_test", "mean"),
            spec_std=("spec_test", "std"),
        )
        .reset_index()
    )

    df_summary = df_summary[df_summary["n_folds_passed"] >= min_folds_passed].copy()

    return df_summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VIC Step 6: CV analysis with strict model selection plus broader threshold curves for selected models."
    )

    # Stage 1: strict model-selection threshold(s)
    ap.add_argument(
        "--spec-min",
        nargs="+",
        type=float,
        default=[0.90],
        help="Specificity threshold(s) used for global model selection.",
    )

    # Stage 2: broader range for selected models only
    ap.add_argument(
        "--spec-range-for-selected",
        nargs="+",
        type=float,
        default=[0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
        help="Specificity threshold(s) used only for reevaluating selected top models.",
    )

    ap.add_argument(
        "--auc-min",
        type=float,
        default=0.70,
        help="Minimum TRAIN AUC required to proceed with threshold scan.",
    )

    # Reporting / selection constraints
    ap.add_argument(
        "--spec-target",
        type=float,
        default=0.0,
        help="Additional minimum TEST specificity requirement. "
             "Use 0.0 to enforce only spec_min on test.",
    )
    ap.add_argument(
        "--sens-target",
        type=float,
        default=0.50,
        help="Minimum mean TEST sensitivity required to keep a model during strict selection.",
    )
    ap.add_argument(
        "--min-folds-passed",
        type=int,
        default=5,
        help="Model must pass (AUC + threshold) in at least this many folds.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top models to keep after strict selection.",
    )
    ap.add_argument(
        "--no-dedup-power",
        action="store_true",
        help="Disable power de-duplication (otherwise keeps best power per identical config).",
    )

    # Output
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="outputs/tables/cv_single",
        help="Output prefix for CSV files.",
    )

    args = ap.parse_args()

    paths = VICPaths.from_env()

    # Load VIC data
    h5_path = paths.h5_dir / "mrc_results.h5"
    labels_path = paths.outputs_dir / "tables" / "case_labels_cv.xlsx"

    df_scores = pd.read_hdf(h5_path, key="mrc/all")
    df_cv = pd.read_excel(labels_path)

    # merge: adds label and fold
    df = df_scores.merge(df_cv[["case_id", "label", "fold", "cohort"]], on="case_id", how="inner")

    # Cache grouped data so we can reuse it in Stage 2
    grouped_data: dict[str, dict] = {}

    group_cols = ["sequence", "shape", "primary_rate", "secondary_rate", "step", "power"]

    for (sequence, shape, pr, sr, step, power), g in df.groupby(group_cols):
        model_id = f"{sequence}|{shape}|PR{pr}|SR{sr}|STEP{step}|PW{power}"

        grouped_data[model_id] = {
            "sequence": sequence,
            "shape": shape,
            "primary_rate": pr,
            "secondary_rate": sr,
            "step": step,
            "power": power,
            "folds": g["fold"].to_numpy().astype(int),
            "labels": g["label"].to_numpy().astype(int),
            "scores": g["MRC_value"].to_numpy().astype(float),
        }

    # ------------------------------------------------------------------
    # Stage 1: strict selection across all models
    # ------------------------------------------------------------------
    fold_rows: list[dict] = []

    for model_id, info in grouped_data.items():
        rows = run_cv_on_scores(
            spec_min_list=list(args.spec_min),
            auc_min=float(args.auc_min),
            folds=info["folds"],
            labels=info["labels"],
            scores=info["scores"],
            model_id=model_id,
        )

        for r in rows:
            r.update(
                {
                    "sequence": info["sequence"],
                    "shape": info["shape"],
                    "primary_rate": info["primary_rate"],
                    "secondary_rate": info["secondary_rate"],
                    "step": info["step"],
                    "power": info["power"],
                }
            )
        fold_rows.extend(rows)

    df_folds = pd.DataFrame(fold_rows)

    fold_path = paths.resolve_output_path(f"{args.out_prefix}_fold_level.csv")
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(fold_path, index=False)

    # Significant models under strict selection rule
    df_sig = summarize_cv_results(
        df_folds,
        min_folds_passed=int(args.min_folds_passed),
        spec_target=float(args.spec_target),
        sens_target=float(args.sens_target),
        dedup_by_power=(not args.no_dedup_power),
    )

    sig_path = paths.resolve_output_path(f"{args.out_prefix}_significant.csv")
    df_sig.to_csv(sig_path, index=False)

    # Top-K selected models
    df_top = (
        df_sig.sort_values(
            by=["sens_mean", "spec_mean", "auc_train_mean"],
            ascending=[False, False, False],
        )
        .head(int(args.top_k))
        .copy()
    )

    top_path = paths.resolve_output_path(f"{args.out_prefix}_top{int(args.top_k)}.csv")
    df_top.to_csv(top_path, index=False)

    selected_model_ids = df_top["model_id"].tolist()

    # ------------------------------------------------------------------
    # Stage 2: reevaluate selected models over a broader spec range
    # ------------------------------------------------------------------
    selected_fold_rows: list[dict] = []

    for model_id in selected_model_ids:
        info = grouped_data[model_id]

        rows = run_cv_on_scores(
            spec_min_list=list(args.spec_range_for_selected),
            auc_min=float(args.auc_min),
            folds=info["folds"],
            labels=info["labels"],
            scores=info["scores"],
            model_id=model_id,
        )

        for r in rows:
            r.update(
                {
                    "sequence": info["sequence"],
                    "shape": info["shape"],
                    "primary_rate": info["primary_rate"],
                    "secondary_rate": info["secondary_rate"],
                    "step": info["step"],
                    "power": info["power"],
                }
            )
        selected_fold_rows.extend(rows)

    df_selected_folds = pd.DataFrame(selected_fold_rows)

    selected_fold_path = paths.resolve_output_path(f"{args.out_prefix}_selected_curve_fold_level.csv")
    df_selected_folds.to_csv(selected_fold_path, index=False)

    df_selected_summary = summarize_selected_curve_results(
        df_selected_folds,
        min_folds_passed=int(args.min_folds_passed),
    )

    selected_summary_path = paths.resolve_output_path(f"{args.out_prefix}_selected_curve_summary.csv")
    df_selected_summary.to_csv(selected_summary_path, index=False)

    print("Saved:")
    print(f" - {fold_path}")
    print(f" - {sig_path}")
    print(f" - {top_path}")
    print(f" - {selected_fold_path}")
    print(f" - {selected_summary_path}")
    print(f"Total fold rows (strict selection stage): {len(df_folds)}")
    print(f"Significant models (strict selection stage): {len(df_sig)}")
    print(f"Selected top models: {len(df_top)}")
    print(f"Selected-curve fold rows: {len(df_selected_folds)}")
    print(f"Selected-curve summary rows: {len(df_selected_summary)}")


if __name__ == "__main__":
    main()
