from __future__ import annotations

import argparse

from vic.paths import VICPaths
from vic.config import load_config
from vic.cv_split import load_case_id_match, make_case_labels, add_stratified_folds, save_labels_cv


def main() -> None:
    ap = argparse.ArgumentParser(description="VIC Step 3: Create labels + stratified CV folds from case_id_match.xlsx.")
    ap.add_argument("--config", type=str, default="config/default.yaml", help="Config YAML path")
    ap.add_argument("--case-id-match", type=str, default="outputs/tables/case_id_match.xlsx", help="Input mapping file")
    ap.add_argument("--out", type=str, default="outputs/tables/case_labels_cv.xlsx", help="Output labels+folds file")
    args = ap.parse_args()

    paths = VICPaths.from_env()
    cfg = load_config(paths.root / args.config)

    df_match = load_case_id_match(paths.resolve_existing_output_or_repo_path(args.case_id_match))
    df_labels = make_case_labels(df_match)
    df_cv = add_stratified_folds(df_labels, n_splits=cfg.n_splits, seed=cfg.seed)

    out_path = save_labels_cv(df_cv, paths.resolve_output_path(args.out))

    print(f"Rows: {len(df_cv)}")
    print(df_cv.head(10))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
