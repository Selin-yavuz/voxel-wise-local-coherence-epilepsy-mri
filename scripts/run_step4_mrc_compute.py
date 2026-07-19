from __future__ import annotations

import argparse

from vic.paths import VICPaths
from vic.config import load_config
from vic.mrc_compute import find_case_pairs, run_batch


def main() -> None:
    ap = argparse.ArgumentParser(description="Run VIC/MRC feature extraction (per-case CSV output).")

    ap.add_argument("--config", type=str, default="config/default.yaml",
                    help="Path to config YAML")

    ap.add_argument("--out", type=str, default="outputs/mrc_results",
                    help="Output folder for per-case CSVs")

    ap.add_argument("--roi-shape", type=str, default=None,
                    help="ROI shape: square or circle (default: use config roi_shapes)")

    ap.add_argument("--primary-rate", type=str, default="all",
                    help="Primary rate: all or 1/2/3")

    ap.add_argument("--secondary-rate", type=str, default="all",
                    help="Secondary rate: all or 1..8")

    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if CSV exists and seems complete")

    args = ap.parse_args()

    paths = VICPaths.from_env()
    cfg = load_config(paths.root / args.config)

    pairs = find_case_pairs(
        ordered_dir=paths.ordered_dir,
        stripped_dir=paths.stripped_dir,
        cohorts=cfg.cohorts,
        sequences=cfg.sequences,
    )

    # parse secondary rate
    if args.secondary_rate == "all":
        sec = "all"
    else:
        sec = int(args.secondary_rate)

    # parse primary rate
    if args.primary_rate == "all":
        pri = "all"
    else:
        pri = int(args.primary_rate)

    # determine ROI shapes
    if args.roi_shape:
        shapes = [args.roi_shape]
    else:
        shapes = cfg.roi_shapes

    total_done = total_skipped = total_failed = 0

    for shape in shapes:

        if shape not in {"square", "circle"}:
            raise ValueError(
                f"Unknown ROI shape: '{shape}'. Supported ROI shapes are 'square' and 'circle'."
            )

        print(f"\nRunning MRC extraction with ROI shape: {shape}")

        stats = run_batch(
            pairs=pairs,
            output_folder=paths.resolve_output_path(args.out),
            original_root=paths.ordered_dir,
            roi_shape=shape,
            primary_rate=pri,
            secondary_rate=sec,
            min_step=cfg.min_step,
            max_step=cfg.max_step,
            min_power=cfg.min_power,
            max_power=cfg.max_power,
            overwrite=args.overwrite,
        )

        total_done += stats["done"]
        total_skipped += stats["skipped"]
        total_failed += stats["failed"]

    print(f"\nFound pairs: {len(pairs)}")
    print(f"Done: {total_done} | Skipped: {total_skipped} | Failed: {total_failed}")


if __name__ == "__main__":
    main()
