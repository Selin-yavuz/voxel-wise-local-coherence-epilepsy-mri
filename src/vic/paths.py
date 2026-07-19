from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by walking upward until pyproject.toml is found."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return here


@dataclass(frozen=True)
class VICPaths:
    """Centralized filesystem paths for the VIC project."""
    root: Path

    @classmethod
    def from_env(cls) -> "VICPaths":
        return cls(root=find_project_root())

    @property
    def data_dir(self) -> Path:
        return Path(os.environ.get("VIC_DATA_DIR", self.root / "data"))

    @property
    def outputs_dir(self) -> Path:
        return Path(os.environ.get("VIC_OUTPUTS_DIR", self.root / "outputs"))

    @property
    def dicom_dir(self) -> Path:
        return self.data_dir / "dicom_data"

    @property
    def nifti_dir(self) -> Path:
        return self.data_dir / "nifti"

    @property
    def ordered_dir(self) -> Path:
        return self.data_dir / "ordered_data"

    @property
    def stripped_dir(self) -> Path:
        return self.data_dir / "stripped_data"

    @property
    def mrc_results_dir(self) -> Path:
        return self.outputs_dir / "mrc_results"

    @property
    def h5_dir(self) -> Path:
        return self.outputs_dir / "h5"

    def resolve_repo_path(self, value: str | Path) -> Path:
        """Resolve an absolute path or a path relative to the repository root."""
        p = Path(value).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.root / p).resolve()

    def resolve_output_path(self, value: str | Path) -> Path:
        """
        Resolve an output path under VIC_OUTPUTS_DIR.

        For backward-compatible command-line defaults, paths beginning with
        "outputs/" are interpreted relative to VIC_OUTPUTS_DIR.
        """
        p = Path(value).expanduser()
        if p.is_absolute():
            return p.resolve()
        parts = p.parts
        if parts and parts[0] == "outputs":
            p = Path(*parts[1:]) if len(parts) > 1 else Path()
        return (self.outputs_dir / p).resolve()

    def resolve_existing_output_or_repo_path(self, value: str | Path) -> Path:
        """
        Resolve a file that may live in VIC_OUTPUTS_DIR or in the repository.

        This keeps older repo-relative paths usable while preferring private
        output folders when VIC_OUTPUTS_DIR is set.
        """
        p = Path(value).expanduser()
        if p.is_absolute():
            return p.resolve()

        output_path = self.resolve_output_path(p)
        if output_path.exists():
            return output_path

        repo_path = self.resolve_repo_path(p)
        if repo_path.exists():
            return repo_path

        return output_path

    def ensure_dirs(self) -> None:
        for p in [
            self.data_dir,
            self.outputs_dir,
            self.mrc_results_dir,
            self.h5_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)
