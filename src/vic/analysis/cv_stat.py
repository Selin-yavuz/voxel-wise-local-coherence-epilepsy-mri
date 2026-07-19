from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def run_cv_on_scores(
    *,
    spec_min_list: list[float],
    auc_min: float,
    folds: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    model_id: str,
) -> list[dict]:
    """
    Core CV routine.
    folds, labels, scores are 1D arrays aligned by case_id.
    """
    rows = []

    for spec_min in spec_min_list:
        for k in np.unique(folds):
            train_mask = folds != k
            test_mask  = folds == k

            y_tr = labels[train_mask]
            s_tr = scores[train_mask]
            y_te = labels[test_mask]
            s_te = scores[test_mask]

            # ---- training AUC filter
            if len(np.unique(y_tr)) < 2:
                auc_tr = np.nan
                passed_auc = False
            else:
                auc_tr = float(roc_auc_score(y_tr, s_tr))
                passed_auc = auc_tr >= auc_min

            thr_star = np.nan
            thr_min  = np.nan
            thr_max  = np.nan
            n_ok     = 0

            sens_te = np.nan
            spec_te = np.nan
            tp = fp = tn = fn = np.nan
            passed_threshold = False

            if passed_auc:
                T_list = np.unique(s_tr)
                T_list.sort()

                ok = []
                for thr in T_list:
                    y_pred_tr = (s_tr >= thr).astype(int)

                    tp_tr = int(((y_tr == 1) & (y_pred_tr == 1)).sum())
                    fp_tr = int(((y_tr == 0) & (y_pred_tr == 1)).sum())
                    tn_tr = int(((y_tr == 0) & (y_pred_tr == 0)).sum())
                    fn_tr = int(((y_tr == 1) & (y_pred_tr == 0)).sum())

                    spec_tr = tn_tr / (tn_tr + fp_tr) if (tn_tr + fp_tr) else 0.0
                    sens_tr = tp_tr / (tp_tr + fn_tr) if (tp_tr + fn_tr) else 0.0

                    if spec_tr >= spec_min:
                        ok.append((float(thr), sens_tr, spec_tr))

                if len(ok) > 0:
                    passed_threshold = True
                    thr_star = ok[0][0]   # smallest threshold meeting constraint
                    thr_min  = ok[0][0]
                    thr_max  = ok[-1][0]
                    n_ok     = len(ok)

                    y_pred_te = (s_te >= thr_star).astype(int)
                    tp = int(((y_te == 1) & (y_pred_te == 1)).sum())
                    fp = int(((y_te == 0) & (y_pred_te == 1)).sum())
                    tn = int(((y_te == 0) & (y_pred_te == 0)).sum())
                    fn = int(((y_te == 1) & (y_pred_te == 0)).sum())

                    sens_te = tp / (tp + fn) if (tp + fn) else 0.0
                    spec_te = tn / (tn + fp) if (tn + fp) else 0.0

            rows.append({
                "spec_min": spec_min,
                "model_id": model_id,
                "fold": int(k),
                "auc_train": auc_tr,
                "passed_auc": bool(passed_auc),
                "passed_threshold": bool(passed_threshold),
                "thr_star": thr_star,
                "thr_min": thr_min,
                "thr_max": thr_max,
                "n_ok_thresholds": int(n_ok),
                "sens_test": sens_te,
                "spec_test": spec_te,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })

    return rows


def summarize_cv_results(
    df_folds: pd.DataFrame,
    *,
    min_folds_passed: int,
    spec_target: float,
    sens_target: float,
    dedup_by_power: bool = True,
) -> pd.DataFrame:
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

    # 1) stability gate
    df_summary = df_summary[df_summary["n_folds_passed"] >= min_folds_passed].copy()

    # 2) quality gate (paper-ready)
    df_summary = df_summary[
        (df_summary["spec_mean"] >= spec_target) &
        (df_summary["sens_mean"] >= sens_target)
    ].copy()

    if df_summary.empty:
        return df_summary

    # 3) keep only one power per identical config
    if dedup_by_power:
        key_cols = ["spec_min", "sequence", "shape", "primary_rate", "secondary_rate", "step"]

        df_summary = df_summary.sort_values(
            by=["sens_mean", "spec_mean", "auc_train_mean", "power"],
            ascending=[False, False, False, True],
        )
        df_summary = df_summary.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

    return df_summary
