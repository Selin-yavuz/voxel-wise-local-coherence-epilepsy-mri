
from __future__ import annotations

import argparse
from pathlib import Path

from vic.paths import VICPaths
from vic.config import load_config
from vic.skullstrip import build_skullstrip_commands, write_shell_script


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate skull-stripping shell script for VIC.")
    ap.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    ap.add_argument("--script-out", type=str, default="outputs/skullstrip_commands.sh", help="Output .sh file path")
    ap.add_argument("--tool", type=str, default="mri_synthstrip", help="Skullstrip tool executable")
    ap.add_argument("--extra-args", type=str, default="--no-csf", help="Extra args for the tool")
    ap.add_argument("--overwrite", action="store_true", help="Include files even if output exists")
    args = ap.parse_args()

    paths = VICPaths.from_env()
    paths.ensure_dirs()
    cfg = load_config(paths.root / args.config)

    commands = build_skullstrip_commands(
        ordered_dir=paths.ordered_dir,
        stripped_dir=paths.stripped_dir,
        cohorts=cfg.cohorts,
        sequences=cfg.sequences,
        overwrite=args.overwrite,
    )

    script_path = paths.resolve_output_path(args.script_out)
    write_shell_script(commands, script_path=script_path, tool=args.tool, extra_args=args.extra_args)

    print(f"Wrote {len(commands)} commands to: {script_path}")


if __name__ == "__main__":
    main()
