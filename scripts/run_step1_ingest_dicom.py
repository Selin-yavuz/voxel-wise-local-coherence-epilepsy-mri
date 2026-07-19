
from __future__ import annotations

import argparse
from pathlib import Path

from vic.paths import VICPaths
from vic.config import load_config
from vic.dicom_ingest import (
    DicomTreeConverter,
    build_case_id_map,
    save_case_id_map_excel,
    copy_to_ordered_data,
    extract_dicom_metadata_table,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="VIC Step 1: DICOM -> NIfTI (if needed) + ordered_data + Excel tables.")
    ap.add_argument("--config", type=str, default="config/default.yaml")
    ap.add_argument("--print-skips", action="store_true", help="Print which folders were skipped (already converted)")
    ap.add_argument("--no-convert", action="store_true", help="Do not run DICOM->NIfTI; only build ordered_data + tables")
    args = ap.parse_args()

    paths = VICPaths.from_env()
    cfg = load_config(paths.root / args.config)

    # 1) build stable case id mapping
    mapping = build_case_id_map(paths.dicom_dir, cfg.cohorts)
    map_path = paths.outputs_dir / "tables" / "case_id_match.xlsx"
    save_case_id_map_excel(mapping, map_path)
    print(f"Wrote: {map_path}")

    # 2) convert DICOM->NIfTI (optional)
    if not args.no_convert:
        conv = DicomTreeConverter()
        conv.transform_files(paths.dicom_dir, paths.nifti_dir, print_skips=args.print_skips)
        print(f"Converted DICOM->NIfTI into: {paths.nifti_dir}")

    # 3) build ordered_data (copy from nifti tree)
    copied = copy_to_ordered_data(paths.nifti_dir, paths.ordered_dir, mapping, cfg.sequences)
    print(f"Copied {copied} NIfTI files into ordered_data: {paths.ordered_dir}")

    # 4) metadata table (optional; requires pydicom)
    df_info = extract_dicom_metadata_table(paths.dicom_dir, mapping, cfg.sequences)
    info_path = paths.outputs_dir / "tables" / "mri_info.xlsx"
    if not df_info.empty:
        info_path.parent.mkdir(parents=True, exist_ok=True)
        df_info.to_excel(info_path, index=False)
        print(f"Wrote: {info_path}")
    else:
        print("No DICOM metadata written (pydicom missing or no readable DICOMs).")


if __name__ == "__main__":
    main()
