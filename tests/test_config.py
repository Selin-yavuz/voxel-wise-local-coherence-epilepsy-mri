from pathlib import Path

from vic.config import load_config


def test_load_default_config() -> None:
    cfg = load_config(Path("config/default.yaml"))

    assert "patients" in cfg.cohorts
    assert "controls" in cfg.cohorts
    assert "t1_tra" in cfg.sequences
    assert cfg.min_step <= cfg.max_step
    assert cfg.min_power <= cfg.max_power

