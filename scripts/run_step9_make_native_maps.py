from __future__ import annotations

import argparse

import pandas as pd

from vic.config import load_config
from vic.mrc_compute import find_case_pairs
from vic.native_maps import (
    NativeMapParams,
    load_thresholds_from_summary,
    make_native_maps_for_case,
    result_to_row,
)
from vic.paths import VICPaths


def case_id_from_path(path) -> str:
    return path.name.replace(".nii.gz", "").replace(".nii", "")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "VIC Step 9: create native-space score maps, threshold masks, "
            "and MRI heatmap overlays for a selected model family."
        )
    )
    ap.add_argument("--config", type=str, default="config/default.yaml")
    ap.add_argument("--sequence", type=str, default="t1_tra")
    ap.add_argument("--shape", type=str, default="circle")
    ap.add_argument("--primary-rate", type=int, default=3)
    ap.add_argument("--secondary-rate", type=int, default=7)
    ap.add_argument("--power", type=int, default=1)
    ap.add_argument("--steps", nargs="+", type=int, default=[3, 4, 5])

    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manual threshold applied to every requested step.",
    )
    ap.add_argument(
        "--threshold-csv",
        type=str,
        default="outputs/tables/cv_single_significant.csv",
        help="CSV containing thr_star_mean values. Used when --threshold is omitted.",
    )
    ap.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Optional case IDs to process, for example P001 P002 C016.",
    )
    ap.add_argument(
        "--cohorts",
        nargs="+",
        default=None,
        help="Optional cohorts to process. Defaults to cohorts in config.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of case pairs to process after filtering.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="native_maps",
        help="Output directory. Relative paths are created under VIC_OUTPUTS_DIR.",
    )
    ap.add_argument(
        "--summary-out",
        type=str,
        default="native_maps/native_map_summary.csv",
        help="Summary CSV path. Relative paths are created under VIC_OUTPUTS_DIR.",
    )
    ap.add_argument("--overlay-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--max-overlay-slices", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument(
        "--no-score-nifti",
        action="store_true",
        help="Do not save the full coherence score map NIfTI.",
    )
    ap.add_argument("--no-mask-nifti", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")

    args = ap.parse_args()

    paths = VICPaths.from_env()
    cfg = load_config(paths.root / args.config)

    steps = [int(s) for s in args.steps]
    if args.threshold is not None:
        thresholds = {step: float(args.threshold) for step in steps}
    else:
        threshold_csv = paths.resolve_existing_output_or_repo_path(args.threshold_csv)
        thresholds = load_thresholds_from_summary(
            threshold_csv,
            sequence=args.sequence,
            shape=args.shape,
            primary_rate=int(args.primary_rate),
            secondary_rate=int(args.secondary_rate),
            power=int(args.power),
            steps=steps,
        )

    cohorts = list(args.cohorts) if args.cohorts else cfg.cohorts
    pairs = find_case_pairs(
        ordered_dir=paths.ordered_dir,
        stripped_dir=paths.stripped_dir,
        cohorts=cohorts,
        sequences=[args.sequence],
    )

    wanted_case_ids = set(args.case_ids) if args.case_ids else None
    if wanted_case_ids is not None:
        pairs = [p for p in pairs if case_id_from_path(p.original_path) in wanted_case_ids]

    if args.limit is not None:
        pairs = pairs[: int(args.limit)]

    if not pairs:
        raise ValueError("No matching case pairs found for requested filters.")

    out_dir = paths.resolve_output_path(args.out_dir)
    summary_out = paths.resolve_output_path(args.summary_out)
    rows: list[dict] = []

    for pair in pairs:
        cohort = pair.original_path.parent.parent.name
        case_id = case_id_from_path(pair.original_path)

        for step in steps:
            params = NativeMapParams(
                sequence=args.sequence,
                shape=args.shape,
                primary_rate=int(args.primary_rate),
                secondary_rate=int(args.secondary_rate),
                step=int(step),
                power=int(args.power),
                threshold=float(thresholds[int(step)]),
            )

            print(f"Processing {case_id} | {params.model_id}")
            result = make_native_maps_for_case(
                original_path=pair.original_path,
                stripped_path=pair.stripped_path,
                cohort=cohort,
                params=params,
                out_dir=out_dir,
                save_coherence_map=not args.no_score_nifti,
                save_mask=not args.no_mask_nifti,
                save_overlays=not args.no_overlays,
                overlay_axis=int(args.overlay_axis),
                max_overlay_slices=int(args.max_overlay_slices),
                dpi=int(args.dpi),
            )
            rows.append(result_to_row(result, outputs_dir=paths.outputs_dir))

    df = pd.DataFrame(rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_out, index=False)

    print("Saved:")
    print(f" - maps: {out_dir}")
    print(f" - summary: {summary_out}")
    print(f"Cases processed: {len({r['case_id'] for r in rows})}")
    print(f"Model-step rows: {len(rows)}")


if __name__ == "__main__":
    main()
