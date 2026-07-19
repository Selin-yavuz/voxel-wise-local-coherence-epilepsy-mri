from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from vic.artificial_data import (
    ArtificialFocusSpec,
    create_artificial_focus_dataset,
    max_finite_coordinate,
)
from vic.feature_extraction import Feature_Extraction


def compute_score_map(
    *,
    original_path: Path,
    stripped_path: Path,
    shape: str,
    primary_rate: int,
    secondary_rate: int,
    step: int,
    power: int,
) -> np.ndarray:
    extractor = Feature_Extraction(
        stripped=str(stripped_path),
        original=str(original_path),
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
    extractor.calculate_features(calculate_MRC=False)
    return extractor.rates[str(secondary_rate)]


def run_artificial_focus_test(
    *,
    root_dir: Path,
    tolerance_voxels: float,
) -> dict[str, object]:
    spec = ArtificialFocusSpec()
    create_artificial_focus_dataset(root_dir, spec=spec)

    original_path = root_dir / "ordered_data" / "patients" / "t2_tra" / "case1.nii.gz"
    stripped_path = root_dir / "stripped_data" / "patients" / "t2_tra" / "case1.nii.gz"

    score_map = compute_score_map(
        original_path=original_path,
        stripped_path=stripped_path,
        shape="circle",
        primary_rate=3,
        secondary_rate=7,
        step=2,
        power=1,
    )
    detected = max_finite_coordinate(score_map)
    distance = float(np.linalg.norm(np.array(detected) - np.array(spec.focus_xyz)))

    passed = distance <= float(tolerance_voxels)
    return {
        "passed": passed,
        "assigned_focus": spec.focus_xyz,
        "detected_focus": detected,
        "distance_voxels": distance,
        "max_score": float(score_map[detected]),
        "root_dir": root_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create artificial NIfTI images and check detection of the assigned focus."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Directory for generated artificial data. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--tolerance-voxels",
        type=float,
        default=0.0,
        help="Maximum allowed distance between assigned and detected focus.",
    )
    args = parser.parse_args()

    if args.root_dir is None:
        with TemporaryDirectory(prefix="vic_artificial_focus_") as tmp:
            result = run_artificial_focus_test(
                root_dir=Path(tmp),
                tolerance_voxels=float(args.tolerance_voxels),
            )
            print_result(result)
            if not result["passed"]:
                raise SystemExit(1)
        return

    result = run_artificial_focus_test(
        root_dir=args.root_dir,
        tolerance_voxels=float(args.tolerance_voxels),
    )
    print_result(result)
    if not result["passed"]:
        raise SystemExit(1)


def print_result(result: dict[str, object]) -> None:
    print(f"Artificial data root: {result['root_dir']}")
    print(f"Assigned focus: {result['assigned_focus']}")
    print(f"Detected focus: {result['detected_focus']}")
    print(f"Distance: {result['distance_voxels']:.3f} voxels")
    print(f"Max score: {result['max_score']:.6g}")
    print(f"Result: {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
