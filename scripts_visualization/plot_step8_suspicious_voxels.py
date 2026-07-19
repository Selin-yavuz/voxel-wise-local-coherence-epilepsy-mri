from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
import pandas as pd


def reshape_arr_for_display(img_arr: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    """
    Same reshape logic used in feature extraction:
    - move smallest axis to axis 0
    - crop empty borders
    Returns:
        reshaped_cropped_array,
        crop_offsets [min0, min1, min2],
        min_slice
    """
    img_shape = np.array(img_arr.shape)
    x, y, z = int(img_shape[0]), int(img_shape[1]), int(img_shape[2])

    min_slice = int(np.argmin(img_shape))

    img_shape[0] = img_shape[min_slice]
    img_shape[min_slice] = x
    if min_slice == 2:
        img_shape[1], img_shape[2] = x, y

    new_arr = np.zeros((img_shape[0], img_shape[1], img_shape[2]), dtype=float)

    for i in range(img_shape[0]):
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

    new_arr = new_arr[min0:max0 + 1, min1:max1 + 1, min2:max2 + 1]
    return new_arr, [int(min0), int(min1), int(min2)], int(min_slice)


def make_model_folder_name(row: pd.Series) -> str:
    return (
        f"{row['sequence']}_{row['shape']}_"
        f"PR{int(row['primary_rate'])}_SR{int(row['secondary_rate'])}_"
        f"PW{int(row['power'])}_STEP{int(row['step'])}"
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_case_id_from_path(path_str: str) -> str:
    p = Path(path_str)
    return p.name.replace(".nii.gz", "").replace(".nii", "")


@dataclass
class VizConfig:
    suspicious_csv: Path
    output_dir: Path
    zoom_margin_factor: float = 3.0
    max_points_per_case: int | None = None
    min_rate_value: float | None = None
    save_zoom: bool = True
    save_slice: bool = True
    annotate_rate: bool = True
    dpi: int = 200


class SuspiciousVoxelVisualizer:
    def __init__(self, config: VizConfig) -> None:
        self.config = config
        self.df = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        if not self.config.suspicious_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.config.suspicious_csv}")

        df = pd.read_csv(self.config.suspicious_csv)

        required = [
            "case_id",
            "sequence",
            "shape",
            "primary_rate",
            "secondary_rate",
            "step",
            "power",
            "rate_value",
            "proc_axis0",
            "proc_axis1",
            "proc_axis2",
            "stripped_path",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if self.config.min_rate_value is not None:
            df = df[df["rate_value"] >= float(self.config.min_rate_value)].copy()

        if df.empty:
            raise ValueError("No suspicious voxel rows left after filtering.")

        return df

    def _take_case_rows(self, df_case: pd.DataFrame) -> pd.DataFrame:
        df_case = df_case.sort_values("rate_value", ascending=False).copy()

        if self.config.max_points_per_case is not None:
            df_case = df_case.head(int(self.config.max_points_per_case)).copy()

        return df_case

    def _load_case_image(self, stripped_path: str) -> np.ndarray:
        img = nib.load(stripped_path)
        data = img.get_fdata()
        reshaped, _, _ = reshape_arr_for_display(data)
        return reshaped

    def _make_roi_patches(self, x: int, y: int, step: int, shape: str) -> list:
        patches_list = []

        if shape == "square":
            roi_len = (step * 2) + 1
            patches_list.append(
                patches.Rectangle(
                    (y - step - 0.5, x - step - 0.5),
                    roi_len,
                    roi_len,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                )
            )

        elif shape == "circle":
            patches_list.append(
                patches.Circle(
                    (y, x),
                    step,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                )
            )

        # center marker
        patches_list.append(
            patches.Circle(
                (y, x),
                0.8,
                facecolor="yellow",
                edgecolor="none",
                alpha=0.5,
            )
        )

        return patches_list

    def _save_slice_figure(
        self,
        img_reshaped: np.ndarray,
        row: pd.Series,
        out_path: Path,
    ) -> None:
        z = int(row["proc_axis0"])
        x = int(row["proc_axis1"])
        y = int(row["proc_axis2"])
        step = int(row["step"])
        shape = str(row["shape"])

        img_slice = img_reshaped[z, :, :]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_slice, cmap="gray", interpolation="none")

        for patch in self._make_roi_patches(x=x, y=y, step=step, shape=shape):
            ax.add_patch(patch)

        title = f"{row['case_id']} | STEP{step} | value={row['rate_value']:.4f}"
        ax.set_title(title)
        ax.set_xlabel("axis2")
        ax.set_ylabel("axis1")

        if self.config.annotate_rate:
            ax.text(
                0.02,
                0.98,
                f"thr={row['threshold']:.4f}\nvalue={row['rate_value']:.4f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
            )

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.config.dpi)
        plt.close(fig)

    def _save_zoom_figure(
        self,
        img_reshaped: np.ndarray,
        row: pd.Series,
        out_path: Path,
    ) -> None:
        z = int(row["proc_axis0"])
        x = int(row["proc_axis1"])
        y = int(row["proc_axis2"])
        step = int(row["step"])
        shape = str(row["shape"])

        img_slice = img_reshaped[z, :, :]
        radius = max(int(np.ceil(step * self.config.zoom_margin_factor)), step + 1)

        x0 = max(0, x - radius)
        x1 = min(img_slice.shape[0], x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(img_slice.shape[1], y + radius + 1)

        zoom_img = img_slice[x0:x1, y0:y1]

        local_x = x - x0
        local_y = y - y0

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            zoom_img,
            cmap="gray",
            interpolation="none",
            vmin=np.nanmin(img_slice),
            vmax=np.nanmax(img_slice),
        )

        for patch in self._make_roi_patches(
            x=local_x,
            y=local_y,
            step=step,
            shape=shape,
        ):
            ax.add_patch(patch)

        ax.set_title(f"{row['case_id']} | zoom | STEP{step}")
        ax.set_xlabel("local axis2")
        ax.set_ylabel("local axis1")

        plt.tight_layout()
        fig.savefig(out_path, dpi=self.config.dpi)
        plt.close(fig)

    def run(self) -> None:
        grouped = self.df.groupby(
            [
                "sequence",
                "shape",
                "primary_rate",
                "secondary_rate",
                "power",
                "step",
                "case_id",
                "stripped_path",
            ],
            dropna=False,
        )

        for keys, df_case in grouped:
            (
                sequence,
                shape,
                primary_rate,
                secondary_rate,
                power,
                step,
                case_id,
                stripped_path,
            ) = keys

            df_case = self._take_case_rows(df_case)
            if df_case.empty:
                continue

            model_folder = ensure_dir(
                self.config.output_dir
                / f"{sequence}_{shape}_PR{int(primary_rate)}_SR{int(secondary_rate)}_PW{int(power)}_STEP{int(step)}"
            )

            img_reshaped = self._load_case_image(stripped_path)

            for idx, (_, row) in enumerate(df_case.iterrows(), start=1):
                base_name = (
                    f"{case_id}"
                    f"_z{int(row['proc_axis0'])}"
                    f"_x{int(row['proc_axis1'])}"
                    f"_y{int(row['proc_axis2'])}"
                    f"_rank{idx:03d}"
                )

                if self.config.save_slice:
                    self._save_slice_figure(
                        img_reshaped=img_reshaped,
                        row=row,
                        out_path=model_folder / f"{base_name}_slice.png",
                    )

                if self.config.save_zoom:
                    self._save_zoom_figure(
                        img_reshaped=img_reshaped,
                        row=row,
                        out_path=model_folder / f"{base_name}_zoom.png",
                    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize suspicious voxels from step8 output and save images per model."
    )
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="CSV from run_step8_extract_suspicious_voxels.py (..._all_voxels.csv)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Folder where images will be saved",
    )
    ap.add_argument(
        "--max-points-per-case",
        type=int,
        default=None,
        help="Optional limit on number of suspicious voxels to save per case, sorted by rate_value descending",
    )
    ap.add_argument(
        "--min-rate-value",
        type=float,
        default=None,
        help="Optional extra rate_value filter before plotting",
    )
    ap.add_argument(
        "--zoom-margin-factor",
        type=float,
        default=3.0,
        help="Zoom radius = step * zoom_margin_factor",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    ap.add_argument(
        "--no-slice",
        action="store_true",
        help="Do not save full-slice overlay images",
    )
    ap.add_argument(
        "--no-zoom",
        action="store_true",
        help="Do not save zoomed images",
    )

    args = ap.parse_args()

    cfg = VizConfig(
        suspicious_csv=Path(args.input).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        zoom_margin_factor=float(args.zoom_margin_factor),
        max_points_per_case=args.max_points_per_case,
        min_rate_value=args.min_rate_value,
        save_zoom=not args.no_zoom,
        save_slice=not args.no_slice,
        dpi=int(args.dpi),
    )

    viz = SuspiciousVoxelVisualizer(cfg)
    viz.run()

    print(f"Saved images under: {cfg.output_dir}")


if __name__ == "__main__":
    main()