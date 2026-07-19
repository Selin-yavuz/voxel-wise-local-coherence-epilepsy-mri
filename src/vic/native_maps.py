from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from vic.feature_extraction import Feature_Extraction, rescale


@dataclass(frozen=True)
class ReshapeMeta:
    original_shape: tuple[int, int, int]
    reoriented_shape: tuple[int, int, int]
    crop_offsets: tuple[int, int, int]
    min_slice: int


@dataclass(frozen=True)
class NativeMapParams:
    sequence: str
    shape: str
    primary_rate: int
    secondary_rate: int
    step: int
    power: int
    threshold: float

    @property
    def model_id(self) -> str:
        return (
            f"{self.sequence}|{self.shape}|PR{self.primary_rate}|"
            f"SR{self.secondary_rate}|STEP{self.step}|PW{self.power}"
        )

    @property
    def folder_name(self) -> str:
        return (
            f"{self.sequence}_{self.shape}_PR{self.primary_rate}_"
            f"SR{self.secondary_rate}_PW{self.power}_STEP{self.step}"
        )


@dataclass(frozen=True)
class NativeMapResult:
    case_id: str
    cohort: str
    params: NativeMapParams
    coherence_map_path: Path | None
    mask_path: Path | None
    overlay_paths: tuple[Path, ...]
    n_finite_voxels: int
    n_threshold_voxels: int
    max_coherence_score: float
    mean_threshold_coherence_score: float


def reshape_arr_with_meta(img_arr: np.ndarray) -> tuple[np.ndarray, ReshapeMeta]:
    original_shape = tuple(int(v) for v in img_arr.shape)
    img_shape = np.array(img_arr.shape)
    x, y, z = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])

    min_slice = int(np.argmin(img_shape))

    img_shape[0] = img_shape[min_slice]
    img_shape[min_slice] = x
    if min_slice == 2:
        img_shape[1], img_shape[2] = x, y

    reoriented_shape = tuple(int(v) for v in img_shape)
    new_arr = np.zeros(reoriented_shape, dtype=float)

    for i in range(reoriented_shape[0]):
        if min_slice == 0:
            new_arr[i, :, :] = img_arr[i, :, :]
        elif min_slice == 1:
            new_arr[i, :, :] = img_arr[:, i, :]
        else:
            new_arr[i, :, :] = img_arr[:, :, i]

    arr_shape = new_arr.shape
    min0 = arr_shape[0]
    min1 = arr_shape[1]
    min2 = arr_shape[2]
    max0 = 0
    max1 = 0
    max2 = 0

    arr_0 = np.zeros((arr_shape[1], arr_shape[2]))
    for i in range(arr_shape[0]):
        min0 = i
        if new_arr[i, :, :].any() != arr_0.any():
            break
    for i in reversed(range(arr_shape[0])):
        max0 = i
        if new_arr[i, :, :].any() != arr_0.any():
            break

    arr_1 = np.zeros((arr_shape[0], arr_shape[2]))
    for i in range(arr_shape[1]):
        min1 = i
        if new_arr[:, i, :].any() != arr_1.any():
            break
    for i in reversed(range(arr_shape[1])):
        max1 = i
        if new_arr[:, i, :].any() != arr_1.any():
            break

    arr_2 = np.zeros((arr_shape[0], arr_shape[1]))
    for i in range(arr_shape[2]):
        min2 = i
        if new_arr[:, :, i].any() != arr_2.any():
            break
    for i in reversed(range(arr_shape[2])):
        max2 = i
        if new_arr[:, :, i].any() != arr_2.any():
            break

    cropped = new_arr[min0:max0 + 1, min1:max1 + 1, min2:max2 + 1]
    meta = ReshapeMeta(
        original_shape=original_shape,
        reoriented_shape=reoriented_shape,
        crop_offsets=(int(min0), int(min1), int(min2)),
        min_slice=int(min_slice),
    )
    return cropped, meta


class FeatureExtractionWithNativeMap(Feature_Extraction):
    def preprocessing_data(self) -> np.ndarray:
        data, self.reshape_meta = reshape_arr_with_meta(self.strp_data)
        data = rescale(data)
        data[data == 0] = np.nan
        del self.org_data, self.strp_data
        return data

    def get_rate_map(self, rate_id: int) -> np.ndarray:
        rate_key = str(rate_id)
        if rate_key not in self.rates:
            raise KeyError(f"Rate '{rate_key}' not found in computed rates.")

        rate_map = self.rates[rate_key]
        if rate_map.ndim != 3:
            raise ValueError(
                f"Expected one 3D rate map, got shape {rate_map.shape}. "
                "Use one primary rate, one power, and one step."
            )
        return rate_map

    def rate_map_to_native_space(self, rate_map: np.ndarray) -> np.ndarray:
        meta: ReshapeMeta = self.reshape_meta
        full = np.full(meta.reoriented_shape, np.nan, dtype=np.float32)

        off0, off1, off2 = meta.crop_offsets
        end0 = off0 + rate_map.shape[0]
        end1 = off1 + rate_map.shape[1]
        end2 = off2 + rate_map.shape[2]
        full[off0:end0, off1:end1, off2:end2] = rate_map.astype(np.float32)

        if meta.min_slice == 0:
            native = full
        elif meta.min_slice == 1:
            native = np.transpose(full, (1, 0, 2))
        elif meta.min_slice == 2:
            native = np.transpose(full, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected min_slice value: {meta.min_slice}")

        if native.shape != meta.original_shape:
            raise ValueError(
                f"Native map shape mismatch: got {native.shape}, "
                f"expected {meta.original_shape}"
            )
        return native.astype(np.float32)


def compute_native_coherence_map(
    *,
    original_path: Path,
    stripped_path: Path,
    params: NativeMapParams,
) -> np.ndarray:
    fe = FeatureExtractionWithNativeMap(
        stripped=str(stripped_path),
        original=str(original_path),
        min_power=params.power,
        max_power=params.power,
        power=params.power,
        min_step=params.step,
        max_step=params.step,
        step=params.step,
        ROI_shape=params.shape,
        primary_rate=params.primary_rate,
        secondary_rate=params.secondary_rate,
        all_powers=False,
        all_steps=False,
    )
    fe.calculate_features(calculate_MRC=False)
    processed_map = fe.get_rate_map(params.secondary_rate)
    return fe.rate_map_to_native_space(processed_map)


def save_coherence_nifti(
    *,
    coherence_map: np.ndarray,
    reference_img: nib.Nifti1Image,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(
        coherence_map.astype(np.float32),
        reference_img.affine,
        reference_img.header,
    )
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))
    return out_path


def save_mask_nifti(
    *,
    coherence_map: np.ndarray,
    threshold: float,
    reference_img: nib.Nifti1Image,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (np.isfinite(coherence_map) & (coherence_map >= float(threshold))).astype(np.uint8)
    img = nib.Nifti1Image(mask, reference_img.affine, reference_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    return out_path


def choose_overlay_slices(
    coherence_map: np.ndarray,
    *,
    threshold: float,
    axis: int,
    max_slices: int,
) -> list[int]:
    mask = np.isfinite(coherence_map) & (coherence_map >= float(threshold))
    reduce_axes = tuple(i for i in range(coherence_map.ndim) if i != int(axis))
    counts = mask.sum(axis=reduce_axes)

    if counts.max() > 0:
        ordered = np.argsort(counts)[::-1]
        return [int(v) for v in ordered[:max_slices] if counts[int(v)] > 0]

    finite_scores = np.where(np.isfinite(coherence_map), coherence_map, -np.inf)
    max_scores = finite_scores.max(axis=reduce_axes)
    if not np.isfinite(max_scores).any():
        return []

    ordered = np.argsort(max_scores)[::-1]
    return [int(v) for v in ordered[:max_slices] if np.isfinite(max_scores[int(v)])]


def take_slice(volume: np.ndarray, *, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError("axis must be 0, 1, or 2")


def save_overlay_pngs(
    *,
    original_data: np.ndarray,
    coherence_map: np.ndarray,
    threshold: float,
    out_dir: Path,
    case_id: str,
    params: NativeMapParams,
    axis: int = 2,
    max_slices: int = 6,
    dpi: int = 180,
) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = choose_overlay_slices(
        coherence_map,
        threshold=threshold,
        axis=axis,
        max_slices=max_slices,
    )
    if not slices:
        return tuple()

    finite_above = coherence_map[
        np.isfinite(coherence_map) & (coherence_map >= float(threshold))
    ]
    finite_all = coherence_map[np.isfinite(coherence_map)]
    if finite_above.size:
        vmax = float(np.nanpercentile(finite_above, 99))
    elif finite_all.size:
        vmax = float(np.nanpercentile(finite_all, 99))
    else:
        vmax = float(threshold) + 1.0
    if not vmax > float(threshold):
        vmax = float(threshold) + 1.0

    saved: list[Path] = []
    for slice_index in slices:
        base_slice = take_slice(original_data, axis=axis, index=slice_index)
        score_slice = take_slice(coherence_map, axis=axis, index=slice_index)
        overlay = np.ma.masked_where(
            ~(np.isfinite(score_slice) & (score_slice >= float(threshold))),
            score_slice,
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.rot90(base_slice), cmap="gray", interpolation="none")
        im = ax.imshow(
            np.rot90(overlay),
            cmap="magma",
            interpolation="none",
            alpha=0.65,
            vmin=float(threshold),
            vmax=vmax,
        )
        ax.set_title(
            f"{case_id} | {params.folder_name} | axis{axis}={slice_index}",
            fontsize=9,
        )
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Coherence score")
        plt.tight_layout()

        out_path = out_dir / f"{case_id}_axis{axis}_{slice_index:03d}_overlay.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        saved.append(out_path)

    return tuple(saved)


def make_native_maps_for_case(
    *,
    original_path: Path,
    stripped_path: Path,
    cohort: str,
    params: NativeMapParams,
    out_dir: Path,
    save_coherence_map: bool = True,
    save_mask: bool = True,
    save_overlays: bool = True,
    overlay_axis: int = 2,
    max_overlay_slices: int = 6,
    dpi: int = 180,
) -> NativeMapResult:
    original_path = Path(original_path)
    stripped_path = Path(stripped_path)
    case_id = original_path.name.replace(".nii.gz", "").replace(".nii", "")

    reference_img = nib.load(str(original_path))
    original_data = reference_img.get_fdata()
    coherence_map = compute_native_coherence_map(
        original_path=original_path,
        stripped_path=stripped_path,
        params=params,
    )

    model_dir = out_dir / params.folder_name / cohort / case_id
    coherence_map_path = None
    mask_path = None
    overlay_paths: tuple[Path, ...] = tuple()

    if save_coherence_map:
        coherence_map_path = save_coherence_nifti(
            coherence_map=coherence_map,
            reference_img=reference_img,
            out_path=model_dir / f"{case_id}_coherence_score_map.nii.gz",
        )

    if save_mask:
        mask_path = save_mask_nifti(
            coherence_map=coherence_map,
            threshold=params.threshold,
            reference_img=reference_img,
            out_path=model_dir / f"{case_id}_mask_thr{params.threshold:.6g}.nii.gz",
        )

    if save_overlays:
        overlay_paths = save_overlay_pngs(
            original_data=original_data,
            coherence_map=coherence_map,
            threshold=params.threshold,
            out_dir=model_dir / "overlays",
            case_id=case_id,
            params=params,
            axis=overlay_axis,
            max_slices=max_overlay_slices,
            dpi=dpi,
        )

    finite_mask = np.isfinite(coherence_map)
    threshold_mask = finite_mask & (coherence_map >= float(params.threshold))
    scores_above = coherence_map[threshold_mask]
    finite_scores = coherence_map[finite_mask]

    return NativeMapResult(
        case_id=case_id,
        cohort=cohort,
        params=params,
        coherence_map_path=coherence_map_path,
        mask_path=mask_path,
        overlay_paths=overlay_paths,
        n_finite_voxels=int(finite_mask.sum()),
        n_threshold_voxels=int(threshold_mask.sum()),
        max_coherence_score=float(np.nanmax(finite_scores)) if finite_scores.size else np.nan,
        mean_threshold_coherence_score=(
            float(np.nanmean(scores_above)) if scores_above.size else np.nan
        ),
    )


def load_thresholds_from_summary(
    summary_csv: Path,
    *,
    sequence: str,
    shape: str,
    primary_rate: int,
    secondary_rate: int,
    power: int,
    steps: Iterable[int],
) -> dict[int, float]:
    df = pd.read_csv(summary_csv)
    required_cols = {
        "sequence",
        "shape",
        "primary_rate",
        "secondary_rate",
        "power",
        "step",
        "thr_star_mean",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in threshold CSV: {sorted(missing)}")

    wanted_steps = {int(s) for s in steps}
    sub = df[
        (df["sequence"] == sequence)
        & (df["shape"] == shape)
        & (df["primary_rate"].astype(int) == int(primary_rate))
        & (df["secondary_rate"].astype(int) == int(secondary_rate))
        & (df["power"].astype(int) == int(power))
        & (df["step"].astype(int).isin(wanted_steps))
    ].copy()

    if sub.empty:
        raise ValueError("No matching threshold rows found for the requested model family.")

    found_steps = set(sub["step"].astype(int).tolist())
    missing_steps = sorted(wanted_steps - found_steps)
    if missing_steps:
        raise ValueError(f"Threshold rows missing for steps: {missing_steps}")

    if sub["thr_star_mean"].isna().any():
        bad_steps = sub.loc[sub["thr_star_mean"].isna(), "step"].astype(int).tolist()
        raise ValueError(f"thr_star_mean contains NaN for steps: {bad_steps}")

    return {
        int(row["step"]): float(row["thr_star_mean"])
        for _, row in sub.sort_values("step").iterrows()
    }


def result_to_row(result: NativeMapResult, *, outputs_dir: Path) -> dict:
    def rel(path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return str(path.relative_to(outputs_dir))
        except ValueError:
            return str(path)

    return {
        "case_id": result.case_id,
        "cohort": result.cohort,
        "sequence": result.params.sequence,
        "shape": result.params.shape,
        "primary_rate": result.params.primary_rate,
        "secondary_rate": result.params.secondary_rate,
        "step": result.params.step,
        "power": result.params.power,
        "threshold": result.params.threshold,
        "model_id": result.params.model_id,
        "n_finite_voxels": result.n_finite_voxels,
        "n_threshold_voxels": result.n_threshold_voxels,
        "max_coherence_score": result.max_coherence_score,
        "mean_threshold_coherence_score": result.mean_threshold_coherence_score,
        "coherence_map_path": rel(result.coherence_map_path),
        "mask_path": rel(result.mask_path),
        "overlay_count": len(result.overlay_paths),
        "overlay_paths": ";".join(rel(path) for path in result.overlay_paths),
    }
