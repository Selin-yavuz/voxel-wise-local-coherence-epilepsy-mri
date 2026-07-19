from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from vic.paths import VICPaths
from vic.config import load_config
from vic.mrc_compute import find_case_pairs
from vic.feature_extraction import Feature_Extraction, rescale


def reshape_arr_with_meta(img_arr: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    """
    Same logic as vic.feature_extraction.reshape_arr(), but also returns min_slice
    so coordinates can be mapped back to original image index space.
    """
    img_shape = np.array(img_arr.shape)
    x, y, z = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])

    min_slice = int(np.argmin(img_shape))

    img_shape[0] = img_shape[min_slice]
    img_shape[min_slice] = x
    if min_slice == 2:
        img_shape[1], img_shape[2] = x, y

    new_arr = np.zeros((img_shape[0], img_shape[1], img_shape[2]), dtype=float)

    for i in range(img_shape[0]):
        if min_slice == 0:
            new_arr[i, :, :] = img_arr[i, :, :]
        elif min_slice == 1:
            new_arr[i, :, :] = img_arr[:, i, :]
        else:
            new_arr[i, :, :] = img_arr[:, :, i]

    arr_shape = new_arr.shape
    min0 = arr_shape[0]
    min1 = arr_shape[1]
    min2 = arr_shape[2]
    max0 = 0
    max1 = 0
    max2 = 0

    arr_0 = np.zeros((arr_shape[1], arr_shape[2]))
    for i in range(arr_shape[0]):
        min0 = i
        if new_arr[i, :, :].any() != arr_0.any():
            break
    for i in reversed(range(arr_shape[0])):
        max0 = i
        if new_arr[i, :, :].any() != arr_0.any():
            break

    arr_1 = np.zeros((arr_shape[0], arr_shape[2]))
    for i in range(arr_shape[1]):
        min1 = i
        if new_arr[:, i, :].any() != arr_1.any():
            break
    for i in reversed(range(arr_shape[1])):
        max1 = i
        if new_arr[:, i, :].any() != arr_1.any():
            break

    arr_2 = np.zeros((arr_shape[0], arr_shape[1]))
    for i in range(arr_shape[2]):
        min2 = i
        if new_arr[:, :, i].any() != arr_2.any():
            break
    for i in reversed(range(arr_shape[2])):
        max2 = i
        if new_arr[:, :, i].any() != arr_2.any():
            break

    new_arr = new_arr[min0:max0 + 1, min1:max1 + 1, min2:max2 + 1]
    return new_arr, [int(min0), int(min1), int(min2)], int(min_slice)


class FeatureExtractionWithCoords(Feature_Extraction):
    """
    Extends Feature_Extraction so we can preserve axis/cropping metadata and
    map suspicious voxel coordinates back to original image index space.
    """

    def preprocessing_data(self) -> np.ndarray:
        data, self.eliminated_coordinates_number, self.min_slice = reshape_arr_with_meta(self.strp_data)
        data = rescale(data)
        data[data == 0] = np.nan
        del self.org_data, self.strp_data
        return data

    def processed_to_original_indices(
        self,
        proc_coord: np.ndarray,
    ) -> tuple[int, int, int]:
        """
        Convert processed/cropped coordinates (axis0, axis1, axis2) back to
        original image index space (orig_i, orig_j, orig_k).
        """
        full0 = int(proc_coord[0]) + int(self.eliminated_coordinates_number[0])
        full1 = int(proc_coord[1]) + int(self.eliminated_coordinates_number[1])
        full2 = int(proc_coord[2]) + int(self.eliminated_coordinates_number[2])

        if self.min_slice == 0:
            orig_i, orig_j, orig_k = full0, full1, full2
        elif self.min_slice == 1:
            orig_i, orig_j, orig_k = full1, full0, full2
        elif self.min_slice == 2:
            orig_i, orig_j, orig_k = full1, full2, full0
        else:
            raise ValueError(f"Unexpected min_slice value: {self.min_slice}")

        return int(orig_i), int(orig_j), int(orig_k)

    def get_rate_map(self, rate_id: int) -> np.ndarray:
        rate_key = str(rate_id)
        if rate_key not in self.rates:
            raise KeyError(f"Rate '{rate_key}' not found in computed rates.")
        rate_map = self.rates[rate_key]
        if rate_map.ndim != 3:
            raise ValueError(
                f"Expected 3D rate map for single-config extraction, got shape {rate_map.shape}"
            )
        return rate_map


def parse_thresholds_from_significant_csv(
    significant_csv: Path,
    *,
    sequence: str,
    shape: str,
    primary_rate: int,
    secondary_rate: int,
    power: int,
    steps: list[int],
) -> dict[int, float]:
    df = pd.read_csv(significant_csv)

    required_cols = [
        "sequence",
        "shape",
        "primary_rate",
        "secondary_rate",
        "power",
        "step",
        "thr_star_mean",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in significant CSV: {missing}")

    sub = df[
        (df["sequence"] == sequence) &
        (df["shape"] == shape) &
        (df["primary_rate"].astype(int) == int(primary_rate)) &
        (df["secondary_rate"].astype(int) == int(secondary_rate)) &
        (df["power"].astype(int) == int(power)) &
        (df["step"].astype(int).isin([int(s) for s in steps]))
    ].copy()

    if sub.empty:
        raise ValueError(
            "No matching rows found in cv_single_significant.csv for the requested model family."
        )

    found_steps = sorted(sub["step"].astype(int).unique().tolist())
    missing_steps = sorted(set(int(s) for s in steps) - set(found_steps))
    if missing_steps:
        raise ValueError(f"Threshold rows missing for steps: {missing_steps}")

    if sub["thr_star_mean"].isna().any():
        bad_steps = sub.loc[sub["thr_star_mean"].isna(), "step"].astype(int).tolist()
        raise ValueError(f"thr_star_mean contains NaN for steps: {bad_steps}")

    return {
        int(r["step"]): float(r["thr_star_mean"])
        for _, r in sub.iterrows()
    }


def extract_case_suspicious_voxels(
    *,
    pair,
    cohort: str,
    sequence: str,
    shape: str,
    primary_rate: int,
    secondary_rate: int,
    power: int,
    step: int,
    threshold: float,
) -> pd.DataFrame:
    fe = FeatureExtractionWithCoords(
        stripped=str(pair.stripped_path),
        original=str(pair.original_path),
        min_power=power,
        max_power=power,
        power=power,
        min_step=step,
        max_step=step,
        step=step,
        ROI_shape=shape,
        primary_rate=primary_rate,
        secondary_rate=secondary_rate,
        all_powers=False,
        all_steps=False,
    )
    fe.calculate_features(calculate_MRC=False)

    rate_map = fe.get_rate_map(rate_id=secondary_rate)

    mask = np.isfinite(rate_map) & (rate_map >= float(threshold))
    coords_proc = np.argwhere(mask)

    case_id = pair.original_path.name.replace(".nii.gz", "").replace(".nii", "")
    orig_img = nib.load(str(pair.original_path))
    affine = orig_img.affine

    rows: list[dict] = []
    for coord in coords_proc:
        proc_axis0, proc_axis1, proc_axis2 = int(coord[0]), int(coord[1]), int(coord[2])
        orig_i, orig_j, orig_k = fe.processed_to_original_indices(coord)

        xyz_mm = nib.affines.apply_affine(
            affine,
            np.array([orig_i, orig_j, orig_k], dtype=float),
        )

        rows.append(
            {
                "case_id": case_id,
                "cohort": cohort,
                "sequence": sequence,
                "shape": shape,
                "primary_rate": int(primary_rate),
                "secondary_rate": int(secondary_rate),
                "step": int(step),
                "power": int(power),
                "threshold": float(threshold),
                "rate_value": float(rate_map[proc_axis0, proc_axis1, proc_axis2]),
                "proc_axis0": proc_axis0,
                "proc_axis1": proc_axis1,
                "proc_axis2": proc_axis2,
                "crop_offset0": int(fe.eliminated_coordinates_number[0]),
                "crop_offset1": int(fe.eliminated_coordinates_number[1]),
                "crop_offset2": int(fe.eliminated_coordinates_number[2]),
                "min_slice": int(fe.min_slice),
                "orig_i": int(orig_i),
                "orig_j": int(orig_j),
                "orig_k": int(orig_k),
                "x_mm": float(xyz_mm[0]),
                "y_mm": float(xyz_mm[1]),
                "z_mm": float(xyz_mm[2]),
                "original_path": str(pair.original_path),
                "stripped_path": str(pair.stripped_path),
            }
        )

    return pd.DataFrame(rows)


def summarize_case_counts(df_voxels: pd.DataFrame) -> pd.DataFrame:
    if df_voxels.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "cohort",
                "sequence",
                "shape",
                "primary_rate",
                "secondary_rate",
                "step",
                "power",
                "threshold",
                "n_suspicious_voxels",
                "max_rate_value",
                "mean_rate_value",
            ]
        )

    return (
        df_voxels.groupby(
            [
                "case_id",
                "cohort",
                "sequence",
                "shape",
                "primary_rate",
                "secondary_rate",
                "step",
                "power",
                "threshold",
            ],
            as_index=False,
        )
        .agg(
            n_suspicious_voxels=("rate_value", "size"),
            max_rate_value=("rate_value", "max"),
            mean_rate_value=("rate_value", "mean"),
        )
        .sort_values(["step", "case_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract all suspicious voxels above per-step thresholds from cv_single_significant.csv."
    )
    ap.add_argument("--config", type=str, default="config/default.yaml")
    ap.add_argument("--sequence", type=str, default="t1_tra")
    ap.add_argument("--shape", type=str, default="circle")
    ap.add_argument("--primary-rate", type=int, default=3)
    ap.add_argument("--secondary-rate", type=int, default=7)
    ap.add_argument("--power", type=int, default=1)
    ap.add_argument("--steps", nargs="+", type=int, default=[3, 4, 5])

    ap.add_argument(
        "--significant-csv",
        type=str,
        default="outputs/tables/cv_single_significant.csv",
        help="CSV file containing thr_star_mean per selected model/step.",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="outputs/tables/suspicious_voxels_focus",
        help="Output prefix for CSV files.",
    )

    args = ap.parse_args()

    paths = VICPaths.from_env()
    cfg = load_config(paths.root / args.config)

    steps = [int(s) for s in args.steps]

    significant_csv = paths.resolve_existing_output_or_repo_path(args.significant_csv)

    threshold_map = parse_thresholds_from_significant_csv(
        significant_csv=significant_csv,
        sequence=args.sequence,
        shape=args.shape,
        primary_rate=int(args.primary_rate),
        secondary_rate=int(args.secondary_rate),
        power=int(args.power),
        steps=steps,
    )

    pairs = find_case_pairs(
        ordered_dir=paths.ordered_dir,
        stripped_dir=paths.stripped_dir,
        cohorts=cfg.cohorts,
        sequences=[args.sequence],
    )

    if not pairs:
        raise ValueError(f"No case pairs found for sequence '{args.sequence}'")

    rows_all: list[pd.DataFrame] = []

    for pair in pairs:
        cohort = pair.original_path.parent.parent.name
        sequence = pair.original_path.parent.name

        for step in steps:
            threshold = float(threshold_map[int(step)])
            df_case = extract_case_suspicious_voxels(
                pair=pair,
                cohort=cohort,
                sequence=sequence,
                shape=args.shape,
                primary_rate=int(args.primary_rate),
                secondary_rate=int(args.secondary_rate),
                power=int(args.power),
                step=int(step),
                threshold=threshold,
            )
            if not df_case.empty:
                rows_all.append(df_case)

    if rows_all:
        df_voxels = pd.concat(rows_all, ignore_index=True)
    else:
        df_voxels = pd.DataFrame(
            columns=[
                "case_id",
                "cohort",
                "sequence",
                "shape",
                "primary_rate",
                "secondary_rate",
                "step",
                "power",
                "threshold",
                "rate_value",
                "proc_axis0",
                "proc_axis1",
                "proc_axis2",
                "crop_offset0",
                "crop_offset1",
                "crop_offset2",
                "min_slice",
                "orig_i",
                "orig_j",
                "orig_k",
                "x_mm",
                "y_mm",
                "z_mm",
                "original_path",
                "stripped_path",
            ]
        )

    df_case_summary = summarize_case_counts(df_voxels)

    out_prefix = paths.resolve_output_path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    voxels_path = Path(f"{out_prefix}_all_voxels.csv")
    case_summary_path = Path(f"{out_prefix}_case_summary.csv")

    df_voxels.to_csv(voxels_path, index=False)
    df_case_summary.to_csv(case_summary_path, index=False)

    print("Saved:")
    print(f" - {voxels_path}")
    print(f" - {case_summary_path}")
    print(f"Selected steps: {steps}")
    print(f"Thresholds from significant CSV: {threshold_map}")
    print(f"Voxel rows: {len(df_voxels)}")
    print(f"Case summary rows: {len(df_case_summary)}")


if __name__ == "__main__":
    main()
