# VIC Epilepsy MRI Pipeline

Reference implementation of a voxel-wise local coherence analysis workflow for
epilepsy MRI research.

This repository contains the code structure for the pipeline. Imaging data and
generated results are not included.

## Code Structure

```text
.
|-- config/
|   `-- default.yaml
|-- scripts/
|   |-- run_step1_ingest_dicom.py
|   |-- run_step2_skullstrip_cmds.py
|   |-- run_step3_make_cv_split.py
|   |-- run_step4_mrc_compute.py
|   |-- run_step5_store_h5.py
|   |-- run_step6_cv_stat.py
|   |-- run_step7_threshold_sweep_focus.py
|   |-- run_step8_extract_suspicious_voxels.py
|   |-- run_step9_make_native_maps.py
|   `-- run_artificial_focus_test.py
|-- scripts_visualization/
|   |-- plot_focus_threshold_sweep.py
|   |-- plot_spec_sens_curves.py
|   `-- plot_step8_suspicious_voxels.py
|-- src/
|   `-- vic/
|       |-- analysis/
|       |   `-- cv_stat.py
|       |-- case_index.py
|       |-- config.py
|       |-- cv_split.py
|       |-- dicom_ingest.py
|       |-- feature_extraction.py
|       |-- h5_store.py
|       |-- mrc_compute.py
|       |-- native_maps.py
|       |-- paths.py
|       |-- artificial_data.py
|       `-- skullstrip.py
|-- tests/
|-- LICENSE
|-- pyproject.toml
`-- README.md
```

## Configuration

`config/default.yaml` defines the default pipeline parameters:

- cohort names;
- MRI sequence names;
- local neighborhood ROI shapes;
- neighborhood step range;
- power transformation range;
- cross-validation fold count and random seed.

## Pipeline Scripts

| Script | Purpose |
| --- | --- |
| `scripts/run_step1_ingest_dicom.py` | Builds case IDs, optionally converts DICOM to NIfTI, organizes NIfTI files, and extracts MRI metadata. |
| `scripts/run_step2_skullstrip_cmds.py` | Generates shell commands for skull stripping. |
| `scripts/run_step3_make_cv_split.py` | Creates labels and stratified subject-level cross-validation folds. |
| `scripts/run_step4_mrc_compute.py` | Computes voxel-wise local coherence / MRC features. |
| `scripts/run_step5_store_h5.py` | Combines per-case MRC feature tables into an HDF5 master table. |
| `scripts/run_step6_cv_stat.py` | Runs cross-validation model selection and summarizes selected parameter configurations. |
| `scripts/run_step7_threshold_sweep_focus.py` | Runs focused threshold sweeps for selected model families. |
| `scripts/run_step8_extract_suspicious_voxels.py` | Extracts threshold-positive candidate voxels. |
| `scripts/run_step9_make_native_maps.py` | Builds native-space coherence maps, threshold masks, and overlay visualizations. |
| `scripts/run_artificial_focus_test.py` | Creates artificial NIfTI images with a known focus and checks whether the score map detects the assigned focus. |

## Visualization Scripts

| Script | Purpose |
| --- | --- |
| `scripts_visualization/plot_focus_threshold_sweep.py` | Plots focused threshold-sweep results. |
| `scripts_visualization/plot_spec_sens_curves.py` | Plots sensitivity/specificity curves. |
| `scripts_visualization/plot_step8_suspicious_voxels.py` | Plots candidate voxels on MRI slices. |

## Source Modules

| Module | Purpose |
| --- | --- |
| `vic.paths` | Centralized project, data, and output path handling. |
| `vic.config` | YAML configuration loading. |
| `vic.dicom_ingest` | DICOM/NIfTI organization, case ID mapping, and metadata extraction. |
| `vic.skullstrip` | Skull-stripping command generation. |
| `vic.cv_split` | Label creation and stratified fold assignment. |
| `vic.feature_extraction` | Local neighborhood feature extraction utilities. |
| `vic.mrc_compute` | Batch local coherence / MRC feature computation. |
| `vic.h5_store` | HDF5 table creation. |
| `vic.analysis.cv_stat` | Cross-validation threshold selection and performance summaries. |
| `vic.native_maps` | Native-space score-map, mask, and overlay generation. |
| `vic.artificial_data` | Artificial NIfTI image generation for reproducible pipeline tests. |

## Cross-validation

For each parameter configuration, each subject is represented by the maximum
voxel-wise coherence score. Parameter configurations are evaluated using
stratified subject-level cross-validation.

Within fold `k`, threshold selection is performed using the training set only:

```text
T*_k = min { T : specificity_train,k(T) >= 0.90 }
```

The selected threshold is then applied unchanged to the held-out test fold.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Artificial Focus Test

The repository includes a deterministic artificial-image test. It creates
synthetic NIfTI volumes with a smooth background and one assigned focal voxel in
the patient image. The test then computes a local coherence score map and checks
whether the maximum-score coordinate matches the assigned focus.

```bash
python scripts/run_artificial_focus_test.py
```

Expected result:

```text
Assigned focus: (2, 20, 20)
Detected focus: (2, 20, 20)
Result: PASS
```

## Run Order

```bash
python scripts/run_step1_ingest_dicom.py --no-convert
python scripts/run_step2_skullstrip_cmds.py
python scripts/run_step3_make_cv_split.py
python scripts/run_step4_mrc_compute.py
python scripts/run_step5_store_h5.py
python scripts/run_step6_cv_stat.py
python scripts/run_step7_threshold_sweep_focus.py
python scripts/run_step8_extract_suspicious_voxels.py
python scripts/run_step9_make_native_maps.py
```

## Tests

```bash
pytest
```
