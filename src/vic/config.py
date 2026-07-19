from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class VICConfig:
    cohorts: list[str]
    sequences: list[str]

    # VIC/MRC parameters
    min_step: int
    max_step: int
    min_power: int
    max_power: int
    roi_shapes: list[str]

    # CV parameters
    n_splits: int
    seed: int


def load_config(path: str | Path) -> VICConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return VICConfig(
        cohorts=list(raw["cohorts"]),
        sequences=list(raw["sequences"]),
        min_step=int(raw["mrc"]["min_step"]),
        max_step=int(raw["mrc"]["max_step"]),
        min_power=int(raw["mrc"]["min_power"]),
        max_power=int(raw["mrc"]["max_power"]),
        roi_shapes=list(raw["mrc"]["roi_shapes"]),
        n_splits=int(raw["cv"]["n_splits"]),
        seed=int(raw["cv"]["seed"]),
    )