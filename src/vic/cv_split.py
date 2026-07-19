from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_case_id_match(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # expected columns from our Step 1: cohort, original_case, case_id
    needed = {"cohort", "case_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"case_id_match missing columns: {missing}")
    return df


def make_case_labels(df_match: pd.DataFrame) -> pd.DataFrame:
    """
    Creates one row per case_id with cohort + label.
    label: patients=1, controls=0
    """
    df = df_match[["cohort", "case_id"]].drop_duplicates().copy()

    def label_from_case_id(cid: str) -> int:
        cid = str(cid)
        if cid.startswith("P"):
            return 1
        if cid.startswith("C"):
            return 0
        # fallback: use cohort
        return 1 if str(df_match.loc[df_match["case_id"] == cid, "cohort"].iloc[0]).lower().startswith("p") else 0

    df["label"] = df["case_id"].apply(label_from_case_id)
    df = df.sort_values(["label", "case_id"]).reset_index(drop=True)
    return df


def add_stratified_folds(df_labels: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Adds fold column using StratifiedKFold on label.
    """
    df = df_labels.copy()
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df["case_id"].values
    y = df["label"].values

    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        df.loc[test_idx, "fold"] = fold_idx

    if (df["fold"] < 0).any():
        raise RuntimeError("Some samples did not get a fold assignment.")

    return df


def save_labels_cv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    return out_path

