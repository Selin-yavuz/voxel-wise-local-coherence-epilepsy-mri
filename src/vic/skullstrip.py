from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SkullstripCommand:
    input_path: Path
    output_path: Path

    def to_shell(self, tool: str = "mri_synthstrip", extra_args: str = "--no-csf") -> str:
        return f'{tool} -i "{self.input_path}" -o "{self.output_path}" {extra_args}'.strip()


def find_ordered_niftis(
    ordered_dir: Path,
    cohorts: Iterable[str],
    sequences: Iterable[str],
) -> list[Path]:
    files: list[Path] = []
    for cohort in cohorts:
        for seq in sequences:
            seq_dir = ordered_dir / cohort / seq
            if not seq_dir.exists():
                continue
            files.extend(sorted(seq_dir.glob("*.nii*")))
    return files


def build_skullstrip_commands(
    ordered_dir: Path,
    stripped_dir: Path,
    cohorts: Iterable[str],
    sequences: Iterable[str],
    overwrite: bool = False,
) -> list[SkullstripCommand]:
    cmds: list[SkullstripCommand] = []
    inputs = find_ordered_niftis(ordered_dir, cohorts, sequences)

    for in_path in inputs:
        cohort = in_path.parents[1].name
        seq = in_path.parents[0].name

        out_path = stripped_dir / cohort / seq / in_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            continue

        cmds.append(
            SkullstripCommand(
                input_path=in_path,
                output_path=out_path,
            )
        )

    return cmds


def write_shell_script(
    commands: list[SkullstripCommand],
    script_path: Path,
    tool: str = "mri_synthstrip",
    extra_args: str = "--no-csf",
) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with script_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for cmd in commands:
            f.write(cmd.to_shell(tool=tool, extra_args=extra_args) + "\n")

