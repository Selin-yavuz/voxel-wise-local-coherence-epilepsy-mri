from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from vic.feature_extraction import (
    Feature_Extraction,
    save_results_to_csv,
    take_csv_file_path,
)


@dataclass(frozen=True)
class CasePair:
    original_path: Path
    stripped_path: Path


def find_case_pairs(
    ordered_dir: Path,
    stripped_dir: Path,
    cohorts: Iterable[str],
    sequences: Iterable[str],
) -> list[CasePair]:
    """
    Matches:
      ordered_dir/{cohort}/{sequence}/{ID}.nii.gz
      stripped_dir/{cohort}/{sequence}/{ID}.nii.gz
    Returns pairs that exist in both locations.
    """
    pairs: list[CasePair] = []

    for cohort in cohorts:
        for seq in sequences:
            orig_seq_dir = ordered_dir / cohort / seq
            strip_seq_dir = stripped_dir / cohort / seq
            if not orig_seq_dir.exists() or not strip_seq_dir.exists():
                continue

            stripped_map = {p.name: p for p in strip_seq_dir.glob("*.nii*")}

            for orig in sorted(orig_seq_dir.glob("*.nii*")):
                strip = stripped_map.get(orig.name)
                if strip is None:
                    continue
                pairs.append(CasePair(original_path=orig, stripped_path=strip))

    return pairs


def check_if_case_config_completed(
    output_folder: Path,
    original_file_path: Path,
    original_root: Path,
    *,
    roi_shape: str,
    min_step: int,
    max_step: int,
    min_power: int,
    max_power: int,
) -> bool:
    """
    Check whether this case CSV already contains rows for the requested ROI shape
    covering the requested step and power ranges.

    This is intentionally conservative for now:
    - it checks shape
    - it checks step range
    - it checks power range

    Since Step 4 usually runs with primary_rate='all' and secondary_rate='all',
    shape + step + power coverage is the important part.
    """
    csv_file_path = take_csv_file_path(output_folder, original_file_path, original_root)
    if not csv_file_path.exists():
        return False

    try:
        df = pd.read_csv(csv_file_path)
    except Exception:
        return False

    needed_cols = {"shape", "step", "power"}
    if df.empty or not needed_cols.issubset(df.columns):
        return False

    sub = df[df["shape"] == roi_shape]
    if sub.empty:
        return False

    has_step_range = (sub["step"].min() <= min_step) and (sub["step"].max() >= max_step)
    has_power_range = (sub["power"].min() <= min_power) and (sub["power"].max() >= max_power)

    return bool(has_step_range and has_power_range)


def run_one_case(
    pair: CasePair,
    output_folder: Path,
    original_root: Path,
    *,
    roi_shape: str,
    primary_rate: str | int,
    secondary_rate: str | int | list[int],
    min_step: int,
    max_step: int,
    min_power: int,
    max_power: int,
    all_steps: bool = True,
    all_powers: bool = True,
) -> Path:
    """
    Runs VIC/MRC feature extraction for one case and appends/saves results to CSV.
    Returns the CSV path.
    """
    fe = Feature_Extraction(
        stripped=str(pair.stripped_path),
        original=str(pair.original_path),
        min_power=min_power,
        max_power=max_power,
        power=min_power,
        min_step=min_step,
        max_step=max_step,
        step=min_step,
        ROI_shape=roi_shape,
        primary_rate=primary_rate,
        secondary_rate=secondary_rate,
        all_powers=all_powers,
        all_steps=all_steps,
    )
    fe.calculate_features(calculate_MRC=True)

    csv_path = save_results_to_csv(
        data=fe.MRC_results,
        output_folder_path=output_folder,
        original_file_path=pair.original_path,
        original_file_root_path=original_root,
    )
    return csv_path


def run_batch(
    pairs: list[CasePair],
    output_folder: Path,
    original_root: Path,
    *,
    roi_shape: str,
    primary_rate: str | int,
    secondary_rate: str | int | list[int],
    min_step: int,
    max_step: int,
    min_power: int,
    max_power: int,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Runs all pairs sequentially (safe + reproducible).
    Returns counts.
    """
    output_folder = Path(output_folder)
    original_root = Path(original_root)
    output_folder.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0

    for pair in pairs:
        try:
            if (not overwrite) and check_if_case_config_completed(
                output_folder=output_folder,
                original_file_path=pair.original_path,
                original_root=original_root,
                roi_shape=roi_shape,
                min_step=min_step,
                max_step=max_step,
                min_power=min_power,
                max_power=max_power,
            ):
                skipped += 1
                continue

            run_one_case(
                pair,
                output_folder=output_folder,
                original_root=original_root,
                roi_shape=roi_shape,
                primary_rate=primary_rate,
                secondary_rate=secondary_rate,
                min_step=min_step,
                max_step=max_step,
                min_power=min_power,
                max_power=max_power,
                all_steps=True,
                all_powers=True,
            )
            done += 1
        except Exception as e:
            failed += 1
            print(f"Failed: {pair.original_path} | {e}")

    return {"done": done, "skipped": skipped, "failed": failed}