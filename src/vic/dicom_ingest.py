from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import dicom2nifti

try:
    import pydicom  # for metadata extraction
except Exception:
    pydicom = None


def create_folder(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


class DicomTreeConverter:
    """
    Recursively mirrors origin_folder into target_folder and converts leaf DICOM series folders to NIfTI.
    Skips leaf folders if target already contains .nii/.nii.gz.
    """

    def take_folder_list(self, folder_path: str | Path) -> list[str]:
        folder_path = str(folder_path)
        folder_list = os.listdir(folder_path)
        folder_list = [f for f in folder_list if os.path.isdir(os.path.join(folder_path, f))]
        return folder_list

    def transform_files(self, origin_folder: str | Path, target_folder: str | Path, print_skips: bool = False) -> None:
        origin_folder = str(origin_folder)
        target_folder = str(target_folder)

        folder_paths_list = self.take_folder_list(origin_folder)

        for folder in folder_paths_list:
            data_path = os.path.join(origin_folder, folder)
            target_path = os.path.join(target_folder, folder)

            new_folder_list = self.take_folder_list(data_path)

            if new_folder_list:
                create_folder(target_path)
                self.transform_files(data_path, target_path, print_skips=print_skips)
            else:
                create_folder(target_path)

                if any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(target_path)):
                    if print_skips:
                        print(f"Skipping (already converted): {target_path}")
                    continue

                dicom2nifti.convert_directory(data_path, target_path)


@dataclass(frozen=True)
class CaseIdMap:
    cohort: str          # patients / controls
    original_case: str   # original folder name under dicom_data/{cohort}/
    case_id: str         # P001 / C001


def build_case_id_map(dicom_root: Path, cohorts: Iterable[str]) -> list[CaseIdMap]:
    """
    Stable mapping by sorting original case folder names.
    patients -> P###
    controls -> C###
    """
    out: list[CaseIdMap] = []
    dicom_root = Path(dicom_root)

    for cohort in cohorts:
        cohort_dir = dicom_root / cohort
        if not cohort_dir.exists():
            continue

        case_folders = sorted([p.name for p in cohort_dir.iterdir() if p.is_dir()])

        prefix = "P" if cohort.lower().startswith("p") else "C"
        for i, case in enumerate(case_folders, start=1):
            out.append(CaseIdMap(cohort=cohort, original_case=case, case_id=f"{prefix}{i:03d}"))

    return out


def save_case_id_map_excel(mapping: list[CaseIdMap], out_path: Path) -> Path:
    df = pd.DataFrame([m.__dict__ for m in mapping])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    return out_path


def copy_to_ordered_data(
    nifti_root: Path,
    ordered_root: Path,
    mapping: list[CaseIdMap],
    sequences: Iterable[str],
) -> int:
    """
    Copies one .nii/.nii.gz from nifti_root/{cohort}/{original_case}/{sequence}/... into:
      ordered_root/{cohort}/{sequence}/{case_id}.nii.gz
    Returns number of files copied.
    """
    copied = 0
    nifti_root = Path(nifti_root)
    ordered_root = Path(ordered_root)

    map_lookup = {(m.cohort, m.original_case): m.case_id for m in mapping}

    for cohort, orig_case in sorted(map_lookup.keys()):
        case_id = map_lookup[(cohort, orig_case)]
        for seq in sequences:
            src_dir = nifti_root / cohort / orig_case / seq
            if not src_dir.exists():
                continue

            # pick first nifti file in that folder (your old code effectively did this)
            nii_files = sorted(list(src_dir.glob("*.nii")) + list(src_dir.glob("*.nii.gz")))
            if not nii_files:
                continue

            src = nii_files[0]
            dst_dir = ordered_root / cohort / seq
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / (case_id + (".nii.gz" if src.suffixes[-2:] == [".nii", ".gz"] else src.suffix))

            if dst.exists():
                continue

            # copy bytes
            dst.write_bytes(src.read_bytes())
            copied += 1

    return copied


def extract_dicom_metadata_table(
    dicom_root: Path,
    mapping: list[CaseIdMap],
    sequences: Iterable[str],
) -> pd.DataFrame:
    """
    Reads one DICOM header per (case, sequence) (first file found).
    Produces a flat table for Excel.

    Requires pydicom. If pydicom is not installed, returns empty df.
    """
    if pydicom is None:
        return pd.DataFrame()

    dicom_root = Path(dicom_root)
    map_lookup = {(m.cohort, m.original_case): m.case_id for m in mapping}

    rows = []
    # common tags you likely care about (safe defaults)
    tags = [
        ("Manufacturer", "Manufacturer"),
        ("ModelName", "ManufacturerModelName"),
        ("MagneticFieldStrength", "MagneticFieldStrength"),
        ("RepetitionTime", "RepetitionTime"),
        ("EchoTime", "EchoTime"),
        ("FlipAngle", "FlipAngle"),
        ("SliceThickness", "SliceThickness"),
        ("PixelSpacing", "PixelSpacing"),
        ("SeriesDescription", "SeriesDescription"),
        ("ProtocolName", "ProtocolName"),
    ]

    for (cohort, orig_case), case_id in sorted(map_lookup.items()):
        for seq in sequences:
            seq_dir = dicom_root / cohort / orig_case / seq
            if not seq_dir.exists():
                continue

            # find a dicom file anywhere under seq_dir
            dcm_files = []
            for root, _, files in os.walk(seq_dir):
                for f in files:
                    if f.lower().endswith(".dcm") or f.isdigit() or "." not in f:
                        dcm_files.append(Path(root) / f)
                if dcm_files:
                    break

            if not dcm_files:
                continue

            dcm_path = dcm_files[0]
            try:
                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
            except Exception:
                continue

            row = {"cohort": cohort, "original_case": orig_case, "case_id": case_id, "sequence": seq}
            for out_name, dicom_tag in tags:
                val = getattr(ds, dicom_tag, None)
                row[out_name] = str(val) if val is not None else ""
            rows.append(row)

    return pd.DataFrame(rows)

