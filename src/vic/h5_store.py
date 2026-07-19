from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MRCRecordFile:
    cohort: str
    sequence: str
    case_id: str
    csv_path: Path


def discover_mrc_csvs(mrc_results_dir: Path, cohorts: Iterable[str], sequences: Iterable[str]) -> list[MRCRecordFile]:
    """
    Expected:
      mrc_results_dir/{cohort}/{sequence}/{case}_MRC.csv
    """
    out: list[MRCRecordFile] = []
    mrc_results_dir = Path(mrc_results_dir)

    for cohort in cohorts:
        for seq in sequences:
            seq_dir = mrc_results_dir / cohort / seq
            if not seq_dir.exists():
                continue
            for csv_path in sorted(seq_dir.glob("*_MRC.csv")):
                case_id = csv_path.name.replace("_MRC.csv", "")
                out.append(MRCRecordFile(cohort=cohort, sequence=seq, case_id=case_id, csv_path=csv_path))

    return out


def load_one_csv(rec: MRCRecordFile) -> pd.DataFrame:
    df = pd.read_csv(rec.csv_path)
    # Add identifiers so the H5 contains everything needed later
    df.insert(0, "case_id", rec.case_id)
    df.insert(0, "sequence", rec.sequence)
    df.insert(0, "cohort", rec.cohort)
    return df


def build_master_table(records: list[MRCRecordFile]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rec in records:
        frames.append(load_one_csv(rec))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_h5_tables(
    df: pd.DataFrame,
    h5_path: Path,
    *,
    key_all: str = "mrc/all",
    key_by_group_prefix: str = "mrc/by_group",
    mode: str = "w",
) -> None:
    """
    Writes:
      - one big table at key_all
      - optional split tables at: {key_by_group_prefix}/{cohort}/{sequence}
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError("No data to write (master table is empty).")

    # HDFStore supports queryable "table" format
    with pd.HDFStore(h5_path, mode=mode, complevel=9, complib="blosc:zstd") as store:
        store.put(key_all, df, format="table", data_columns=True)

        # also store per cohort/sequence for faster access later
        for (cohort, sequence), sub in df.groupby(["cohort", "sequence"], sort=False):
            key = f"{key_by_group_prefix}/{cohort}/{sequence}"
            store.put(key, sub, format="table", data_columns=True)

