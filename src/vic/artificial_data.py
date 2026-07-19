from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class ArtificialFocusSpec:
    """Settings for deterministic artificial NIfTI focus data."""

    shape: tuple[int, int, int] = (5, 40, 40)
    focus_xyz: tuple[int, int, int] = (2, 20, 20)
    focus_delta: float = 8.0
    n_cases_per_cohort: int = 5
    sequences: tuple[str, ...] = ("t2_tra", "t2_cor")


def create_artificial_focus_volume(
    *,
    spec: ArtificialFocusSpec = ArtificialFocusSpec(),
    include_focus: bool,
) -> np.ndarray:
    """
    Create a smooth nonzero 3D image with an optional assigned focal outlier.

    The nonzero background avoids degenerate percentile rescaling. The patient
    variant adds one high-intensity voxel at ``spec.focus_xyz``.
    """
    z, y, x = np.indices(spec.shape, dtype=np.float32)
    data = 10.0 + 0.20 * z + 0.07 * y + 0.03 * x
    data += 0.15 * np.sin(y / 4.0) + 0.12 * np.cos(x / 5.0)

    if include_focus:
        data[spec.focus_xyz] += float(spec.focus_delta)

    return data.astype(np.float32)


def save_artificial_nifti(data: np.ndarray, path: Path) -> Path:
    """Save one artificial volume as a NIfTI image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    img.set_data_dtype(np.float32)
    nib.save(img, str(path))
    return path


def create_artificial_focus_dataset(
    root_dir: Path,
    *,
    spec: ArtificialFocusSpec = ArtificialFocusSpec(),
    data_kinds: Iterable[str] = ("ordered_data", "stripped_data"),
) -> dict[str, object]:
    """
    Create artificial ordered/stripped NIfTI trees for pipeline tests.

    Layout:
      root_dir/{ordered_data,stripped_data}/{patients,controls}/{sequence}/case1.nii.gz

    Returns metadata including the assigned focus coordinate.
    """
    root_dir = Path(root_dir)
    written: list[Path] = []

    for data_kind in data_kinds:
        for cohort, include_focus in (("controls", False), ("patients", True)):
            volume = create_artificial_focus_volume(spec=spec, include_focus=include_focus)
            for sequence in spec.sequences:
                for case_idx in range(1, spec.n_cases_per_cohort + 1):
                    path = (
                        root_dir
                        / data_kind
                        / cohort
                        / sequence
                        / f"case{case_idx}.nii.gz"
                    )
                    written.append(save_artificial_nifti(volume, path))

    return {
        "root_dir": root_dir,
        "focus_xyz": spec.focus_xyz,
        "n_files": len(written),
        "files": tuple(written),
    }


def max_finite_coordinate(score_map: np.ndarray) -> tuple[int, int, int]:
    """Return the coordinate of the highest finite value in a 3D score map."""
    finite = np.where(np.isfinite(score_map), score_map, -np.inf)
    if not np.isfinite(finite).any():
        raise ValueError("score_map contains no finite values")
    return tuple(int(v) for v in np.unravel_index(np.argmax(finite), finite.shape))
