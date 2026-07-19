from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vic.paths import VICPaths


@dataclass
class ModelSweepConfig:
    sequence: str
    shape: str
    primary_rate: int
    secondary_rate: int
    power: int
    steps: list[int]
    n_thresholds: int
    top10_input: str
    clip_to_score_range: bool = False


def model_id_from_parts(
    sequence: str,
    shape: str,
    primary_rate: int,
    secondary_rate: int,
    step: int,
    power: int,
) -> str:
    return f"{sequence}|{shape}|PR{primary_rate}|SR{secondary_rate}|STEP{step}|PW{power}"


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sens": float(sens) if not np.isnan(sens) else np.nan,
        "spec": float(spec) if not np.isnan(spec) else np.nan,
    }


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


class FocusThresholdSweep:
    def __init__(self, config: ModelSweepConfig) -> None:
        self.config = config
        self.paths = VICPaths.from_env()

    def load_merged_data(self) -> pd.DataFrame:
        h5_path = self.paths.h5_dir / "mrc_results.h5"
        labels_path = self.paths.outputs_dir / "tables" / "case_labels_cv.xlsx"

        df_scores = pd.read_hdf(h5_path, key="mrc/all")
        df_cv = pd.read_excel(labels_path)

        df = df_scores.merge(
            df_cv[["case_id", "label", "fold", "cohort"]],
            on="case_id",
            how="inner",
        )
        return df

    def load_top10_data(self) -> pd.DataFrame:
        top10_path = self.paths.resolve_existing_output_or_repo_path(self.config.top10_input)
        if not top10_path.exists():
            raise FileNotFoundError(f"Top10 file not found: {top10_path}")

        df_top = pd.read_csv(top10_path)
        required_cols = [
            "sequence",
            "shape",
            "primary_rate",
            "secondary_rate",
            "step",
            "power",
            "thr_star_mean",
        ]
        missing = [c for c in required_cols if c not in df_top.columns]
        if missing:
            raise ValueError(f"Missing required columns in top10 file: {missing}")

        return df_top

    def filter_selected_models(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        mask = (
            (df["sequence"] == cfg.sequence) &
            (df["shape"] == cfg.shape) &
            (df["primary_rate"] == cfg.primary_rate) &
            (df["secondary_rate"] == cfg.secondary_rate) &
            (df["power"] == cfg.power) &
            (df["step"].isin(cfg.steps))
        )

        out = df.loc[mask].copy()

        if out.empty:
            raise ValueError(
                "No rows found for requested model family in merged data: "
                f"sequence={cfg.sequence}, shape={cfg.shape}, "
                f"primary_rate={cfg.primary_rate}, secondary_rate={cfg.secondary_rate}, "
                f"power={cfg.power}, steps={cfg.steps}"
            )

        found_steps = sorted(out["step"].astype(int).unique().tolist())
        missing_steps = sorted(set(cfg.steps) - set(found_steps))
        if missing_steps:
            raise ValueError(
                f"Some requested steps were not found in merged data: {missing_steps}. "
                f"Found steps: {found_steps}"
            )

        out["model_id"] = out.apply(
            lambda r: model_id_from_parts(
                sequence=str(r["sequence"]),
                shape=str(r["shape"]),
                primary_rate=int(r["primary_rate"]),
                secondary_rate=int(r["secondary_rate"]),
                step=int(r["step"]),
                power=int(r["power"]),
            ),
            axis=1,
        )

        return out

    def filter_top10_threshold_rows(self, df_top: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        mask = (
            (df_top["sequence"] == cfg.sequence) &
            (df_top["shape"] == cfg.shape) &
            (df_top["primary_rate"] == cfg.primary_rate) &
            (df_top["secondary_rate"] == cfg.secondary_rate) &
            (df_top["power"] == cfg.power) &
            (df_top["step"].isin(cfg.steps))
        )

        out = df_top.loc[mask].copy()

        if out.empty:
            raise ValueError(
                "No rows found for requested model family in top10 file: "
                f"sequence={cfg.sequence}, shape={cfg.shape}, "
                f"primary_rate={cfg.primary_rate}, secondary_rate={cfg.secondary_rate}, "
                f"power={cfg.power}, steps={cfg.steps}"
            )

        found_steps = sorted(out["step"].astype(int).unique().tolist())
        missing_steps = sorted(set(cfg.steps) - set(found_steps))
        if missing_steps:
            raise ValueError(
                f"Some requested steps were not found in top10 file: {missing_steps}. "
                f"Found steps: {found_steps}"
            )

        if out["thr_star_mean"].isna().any():
            bad_steps = out.loc[out["thr_star_mean"].isna(), "step"].astype(int).tolist()
            raise ValueError(f"thr_star_mean contains NaN for step(s): {bad_steps}")

        return out.sort_values("step").reset_index(drop=True)

    def build_threshold_grid(
        self,
        df_top_sel: pd.DataFrame,
        df_selected: pd.DataFrame,
    ) -> tuple[np.ndarray, dict]:
        thr_means = df_top_sel["thr_star_mean"].to_numpy(dtype=float)

        m_min = float(np.min(thr_means))
        m_max = float(np.max(thr_means))
        d = float(m_max - m_min)

        if np.isclose(d, 0.0):
            eps = max(abs(m_min) * 0.05, 1e-6)
            lower = m_min - eps
            upper = m_max + eps
        else:
            lower = m_min - d
            upper = m_max + d

        score_min = float(df_selected["MRC_value"].min())
        score_max = float(df_selected["MRC_value"].max())

        unclipped_lower = lower
        unclipped_upper = upper

        if self.config.clip_to_score_range:
            lower = max(lower, score_min)
            upper = min(upper, score_max)

        if not upper > lower:
            raise ValueError(
                f"Invalid threshold interval after processing: lower={lower}, upper={upper}"
            )

        thresholds = np.linspace(lower, upper, self.config.n_thresholds)

        meta = {
            "threshold_mode": "from_top10_thr_star_mean_range_expansion",
            "n_thresholds": int(self.config.n_thresholds),
            "thr_star_mean_min": m_min,
            "thr_star_mean_max": m_max,
            "thr_star_mean_range": d,
            "threshold_lower_unclipped": unclipped_lower,
            "threshold_upper_unclipped": unclipped_upper,
            "threshold_lower_final": lower,
            "threshold_upper_final": upper,
            "score_min_selected_models": score_min,
            "score_max_selected_models": score_max,
            "clip_to_score_range": bool(self.config.clip_to_score_range),
            "n_top10_rows_used": int(len(df_top_sel)),
        }
        return thresholds, meta

    def run_fold_sweep(
        self,
        df_selected: pd.DataFrame,
        thresholds: np.ndarray,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        group_cols = [
            "sequence",
            "shape",
            "primary_rate",
            "secondary_rate",
            "step",
            "power",
            "model_id",
        ]

        for keys, g in df_selected.groupby(group_cols):
            sequence, shape, pr, sr, step, power, model_id = keys

            folds = np.sort(g["fold"].astype(int).unique())
            labels = g["label"].to_numpy(dtype=int)
            scores = g["MRC_value"].to_numpy(dtype=float)
            fold_ids = g["fold"].to_numpy(dtype=int)

            for fold in folds:
                train_mask = fold_ids != fold
                test_mask = fold_ids == fold

                y_tr = labels[train_mask]
                s_tr = scores[train_mask]
                y_te = labels[test_mask]
                s_te = scores[test_mask]

                auc_tr = safe_auc(y_tr, s_tr)
                auc_te = safe_auc(y_te, s_te)

                for idx, thr in enumerate(thresholds, start=1):
                    train_m = compute_metrics(y_tr, s_tr, float(thr))
                    test_m = compute_metrics(y_te, s_te, float(thr))

                    rows.append(
                        {
                            "sequence": sequence,
                            "shape": shape,
                            "primary_rate": int(pr),
                            "secondary_rate": int(sr),
                            "step": int(step),
                            "power": int(power),
                            "model_id": model_id,
                            "fold": int(fold),
                            "threshold_index": int(idx),
                            "threshold": float(thr),
                            "auc_train": auc_tr,
                            "auc_test": auc_te,
                            "tp_train": train_m["tp"],
                            "fp_train": train_m["fp"],
                            "tn_train": train_m["tn"],
                            "fn_train": train_m["fn"],
                            "sens_train": train_m["sens"],
                            "spec_train": train_m["spec"],
                            "tp_test": test_m["tp"],
                            "fp_test": test_m["fp"],
                            "tn_test": test_m["tn"],
                            "fn_test": test_m["fn"],
                            "sens_test": test_m["sens"],
                            "spec_test": test_m["spec"],
                        }
                    )

        if not rows:
            raise ValueError("No fold-level rows were created.")

        return pd.DataFrame(rows)

    def summarize(self, df_folds: pd.DataFrame) -> pd.DataFrame:
        group_cols = [
            "sequence",
            "shape",
            "primary_rate",
            "secondary_rate",
            "step",
            "power",
            "model_id",
            "threshold_index",
            "threshold",
        ]

        df_summary = (
            df_folds.groupby(group_cols, as_index=False)
            .agg(
                n_folds=("fold", "nunique"),
                auc_train_mean=("auc_train", "mean"),
                auc_train_std=("auc_train", "std"),
                auc_test_mean=("auc_test", "mean"),
                auc_test_std=("auc_test", "std"),
                sens_train_mean=("sens_train", "mean"),
                sens_train_std=("sens_train", "std"),
                spec_train_mean=("spec_train", "mean"),
                spec_train_std=("spec_train", "std"),
                sens_test_mean=("sens_test", "mean"),
                sens_test_std=("sens_test", "std"),
                spec_test_mean=("spec_test", "mean"),
                spec_test_std=("spec_test", "std"),
                tp_test_sum=("tp_test", "sum"),
                fp_test_sum=("fp_test", "sum"),
                tn_test_sum=("tn_test", "sum"),
                fn_test_sum=("fn_test", "sum"),
            )
            .sort_values(["step", "threshold_index"])
            .reset_index(drop=True)
        )

        return df_summary

    def threshold_meta_table(
        self,
        df_selected: pd.DataFrame,
        df_top_sel: pd.DataFrame,
        thresholds: np.ndarray,
        meta: dict,
    ) -> pd.DataFrame:
        per_step_top10 = (
            df_top_sel.groupby("step", as_index=False)
            .agg(
                thr_star_mean=("thr_star_mean", "mean"),
                thr_star_min=("thr_star_mean", "min"),
                thr_star_max=("thr_star_mean", "max"),
                n_rows=("thr_star_mean", "size"),
            )
            .sort_values("step")
            .reset_index(drop=True)
        )

        per_step_scores = (
            df_selected.groupby("step", as_index=False)
            .agg(
                mrc_mean=("MRC_value", "mean"),
                mrc_std=("MRC_value", "std"),
                mrc_min=("MRC_value", "min"),
                mrc_max=("MRC_value", "max"),
                n_cases=("case_id", "nunique"),
            )
            .sort_values("step")
            .reset_index(drop=True)
        )

        df_step = per_step_top10.merge(per_step_scores, on="step", how="outer")

        meta_rows = [
            {"section": "global", "step": np.nan, "name": k, "value": v}
            for k, v in meta.items()
        ]

        threshold_rows = [
            {
                "section": "threshold_grid",
                "step": np.nan,
                "name": f"threshold_{i+1}",
                "value": float(t),
            }
            for i, t in enumerate(thresholds)
        ]

        step_rows = []
        for _, r in df_step.iterrows():
            step_val = int(r["step"])
            for key in [
                "thr_star_mean",
                "thr_star_min",
                "thr_star_max",
                "n_rows",
                "mrc_mean",
                "mrc_std",
                "mrc_min",
                "mrc_max",
                "n_cases",
            ]:
                step_rows.append(
                    {
                        "section": "per_step_stats",
                        "step": step_val,
                        "name": key,
                        "value": float(r[key]) if pd.notna(r[key]) else np.nan,
                    }
                )

        return pd.DataFrame(meta_rows + threshold_rows + step_rows)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.load_merged_data()
        df_selected = self.filter_selected_models(df)

        df_top = self.load_top10_data()
        df_top_sel = self.filter_top10_threshold_rows(df_top)

        thresholds, meta = self.build_threshold_grid(df_top_sel, df_selected)

        df_folds = self.run_fold_sweep(df_selected, thresholds)
        df_summary = self.summarize(df_folds)
        df_meta = self.threshold_meta_table(df_selected, df_top_sel, thresholds, meta)

        return df_folds, df_summary, df_meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Focused threshold sweep using thr_star_mean from cv_single_top10."
    )
    ap.add_argument("--sequence", type=str, default="t1_tra")
    ap.add_argument("--shape", type=str, default="circle")
    ap.add_argument("--primary-rate", type=int, default=3)
    ap.add_argument("--secondary-rate", type=int, default=7)
    ap.add_argument("--power", type=int, default=1)
    ap.add_argument("--steps", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--n-thresholds", type=int, default=20)
    ap.add_argument(
        "--top10-input",
        type=str,
        default="outputs/tables/cv_single_top10.csv",
        help="Input top10 CSV containing thr_star_mean.",
    )
    ap.add_argument(
        "--clip-to-score-range",
        action="store_true",
        help="Clip threshold bounds to observed MRC score min/max.",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="outputs/tables/focus_threshold_sweep",
        help="Output prefix for CSV files.",
    )

    args = ap.parse_args()

    cfg = ModelSweepConfig(
        sequence=args.sequence,
        shape=args.shape,
        primary_rate=int(args.primary_rate),
        secondary_rate=int(args.secondary_rate),
        power=int(args.power),
        steps=list(args.steps),
        n_thresholds=int(args.n_thresholds),
        top10_input=args.top10_input,
        clip_to_score_range=bool(args.clip_to_score_range),
    )

    runner = FocusThresholdSweep(cfg)
    df_folds, df_summary, df_meta = runner.run()

    paths = VICPaths.from_env()
    out_prefix = paths.resolve_output_path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    folds_path = Path(f"{out_prefix}_fold_level.csv")
    summary_path = Path(f"{out_prefix}_summary.csv")
    meta_path = Path(f"{out_prefix}_threshold_meta.csv")

    df_folds.to_csv(folds_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_meta.to_csv(meta_path, index=False)

    print("Saved:")
    print(f" - {folds_path}")
    print(f" - {summary_path}")
    print(f" - {meta_path}")
    print(f"Selected steps: {cfg.steps}")
    print(f"Fold-level rows: {len(df_folds)}")
    print(f"Summary rows: {len(df_summary)}")
    print(f"Meta rows: {len(df_meta)}")


if __name__ == "__main__":
    main()
