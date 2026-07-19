
from __future__ import annotations

import argparse
from pathlib import Path

from vic.paths import VICPaths
from vic.config import load_config
from vic.h5_store import discover_mrc_csvs, build_master_table, write_h5_tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Store all per-case MRC CSVs into an HDF5 file (raw rows).")
    ap.add_argument("--config", type=str, default="config/default.yaml", help="Config YAML path")
    ap.add_argument("--mrc-dir", type=str, default="outputs/mrc_results", help="Folder containing per-case CSVs")
    ap.add_argument("--out", type=str, default="outputs/h5/mrc_results.h5", help="Output HDF5 path")
    args = ap.parse_args()

    paths = VICPaths.from_env()
    cfg = load_config(paths.root / args.config)

    mrc_dir = paths.resolve_existing_output_or_repo_path(args.mrc_dir)
    out_h5 = paths.resolve_output_path(args.out)

    records = discover_mrc_csvs(mrc_dir, cfg.cohorts, cfg.sequences)
    print(f"Found {len(records)} CSV files under {mrc_dir}")

    df = build_master_table(records)
    print(f"Combined rows: {len(df)} | columns: {list(df.columns)}")

    write_h5_tables(df, out_h5)
    print(f"Wrote HDF5 to: {out_h5}")


if __name__ == "__main__":
    main()
