from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_case_index(ordered_dir: Path, stripped_dir: Path | None = None) -> pd.DataFrame:
    rows = []
    ordered_dir = Path(ordered_dir)
    stripped_dir = Path(stripped_dir) if stripped_dir is not None else None

    for cohort_dir in sorted([p for p in ordered_dir.iterdir() if p.is_dir()]):
        cohort = cohort_dir.name
        for seq_dir in sorted([p for p in cohort_dir.iterdir() if p.is_dir()]):
            sequence = seq_dir.name
            for f in sorted(seq_dir.glob("*.nii*")):
                case_id = f.name.replace(".nii.gz", "").replace(".nii", "")
                row = {
                    "cohort": cohort,
                    "sequence": sequence,
                    "case_id": case_id,
                    "ordered_path": str(f),
                }
                if stripped_dir is not None:
                    sp = stripped_dir / cohort / sequence / f.name
                    row["stripped_path"] = str(sp) if sp.exists() else ""
                rows.append(row)

    df = pd.DataFrame(rows)
    # useful ordering
    if not df.empty:
        df = df.sort_values(["cohort", "sequence", "case_id"]).reset_index(drop=True)
    return df


def save_case_index(df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in [".xlsx"]:
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path

