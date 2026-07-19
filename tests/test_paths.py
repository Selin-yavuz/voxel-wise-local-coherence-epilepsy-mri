from pathlib import Path

from vic.paths import VICPaths


def test_resolve_output_path_strips_legacy_outputs_prefix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VIC_OUTPUTS_DIR", str(tmp_path / "private_outputs"))
    paths = VICPaths(root=Path("/repo"))

    resolved = paths.resolve_output_path("outputs/tables/cv_single_top10.csv")

    assert resolved == tmp_path / "private_outputs" / "tables" / "cv_single_top10.csv"


def test_resolve_existing_output_or_repo_path_prefers_outputs_dir(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "private_outputs"
    output_file = output_dir / "tables" / "case_labels_cv.xlsx"
    output_file.parent.mkdir(parents=True)
    output_file.touch()

    monkeypatch.setenv("VIC_OUTPUTS_DIR", str(output_dir))
    paths = VICPaths(root=tmp_path / "repo")

    resolved = paths.resolve_existing_output_or_repo_path("outputs/tables/case_labels_cv.xlsx")

    assert resolved == output_file.resolve()
