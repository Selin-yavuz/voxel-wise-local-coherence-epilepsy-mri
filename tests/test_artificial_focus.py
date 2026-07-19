from __future__ import annotations

import numpy as np

from vic.artificial_data import (
    ArtificialFocusSpec,
    create_artificial_focus_dataset,
    max_finite_coordinate,
)
from vic.feature_extraction import Feature_Extraction


def test_create_artificial_focus_dataset_writes_expected_nifti_tree(tmp_path) -> None:
    spec = ArtificialFocusSpec(n_cases_per_cohort=2, sequences=("t2_tra",))

    metadata = create_artificial_focus_dataset(tmp_path, spec=spec)

    assert metadata["focus_xyz"] == spec.focus_xyz
    assert metadata["n_files"] == 8
    assert (tmp_path / "ordered_data" / "patients" / "t2_tra" / "case1.nii.gz").exists()
    assert (tmp_path / "ordered_data" / "controls" / "t2_tra" / "case1.nii.gz").exists()
    assert (tmp_path / "stripped_data" / "patients" / "t2_tra" / "case1.nii.gz").exists()
    assert (tmp_path / "stripped_data" / "controls" / "t2_tra" / "case1.nii.gz").exists()


def test_artificial_patient_focus_is_highest_score_voxel(tmp_path) -> None:
    spec = ArtificialFocusSpec(n_cases_per_cohort=1, sequences=("t2_tra",))
    create_artificial_focus_dataset(tmp_path, spec=spec)

    original_path = tmp_path / "ordered_data" / "patients" / "t2_tra" / "case1.nii.gz"
    stripped_path = tmp_path / "stripped_data" / "patients" / "t2_tra" / "case1.nii.gz"

    extractor = Feature_Extraction(
        stripped=str(stripped_path),
        original=str(original_path),
        min_power=1,
        max_power=1,
        power=1,
        min_step=2,
        max_step=2,
        step=2,
        ROI_shape="circle",
        primary_rate=3,
        secondary_rate=7,
        all_powers=False,
        all_steps=False,
    )
    extractor.calculate_features(calculate_MRC=False)

    score_map = extractor.rates["7"]
    detected = max_finite_coordinate(score_map)
    distance = float(np.linalg.norm(np.array(detected) - np.array(spec.focus_xyz)))

    assert detected == spec.focus_xyz
    assert distance == 0.0
    assert np.isfinite(score_map[detected])
