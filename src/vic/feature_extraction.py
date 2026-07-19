from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Union

import numpy as np
import pandas as pd
import nibabel as nib


# =========================
# Small helpers (minimal set)
# =========================

def concat_2_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if not df1.empty and not df2.empty:
        return pd.concat([df1, df2], axis=0, ignore_index=True)
    if not df1.empty:
        return df1.copy()
    if not df2.empty:
        return df2.copy()
    return pd.DataFrame()


def make_sequence(value_max: int = 0, value_min: int = 0) -> np.ndarray:
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max")
    return np.arange(value_min, value_max + 1)


def take_rate_relation_list() -> list[list[str]]:
    rates = []
    rates.append(["2", "1", "0"])
    rates.append(["3", "1", "1"])
    rates.append(["4", "1", "2"])
    rates.append(["5", "2", "1"])
    rates.append(["6", "2", "2"])
    rates.append(["7", "3", "2"])
    rates.append(["8", "5", "2"])
    return rates


def take_pixel_numbers_of_kernel(kernel_len: int, shape: str = "square") -> int:
    if shape == "square":
        return int(np.square(kernel_len))
    if shape == "circle":
        r = int((kernel_len - 1) / 2)
        pixel_number = np.sum(
            [1 for i in range(-r, r + 1) for j in range(-r, r + 1) if (i**2 + j**2) <= r**2]
        )
        return int(pixel_number)

    raise ValueError(
        f"Unknown ROI shape: '{shape}'. Supported ROI shapes are 'square' and 'circle'."
    )


def take_positive_minimum(data: np.ndarray) -> float:
    tmp = np.array(data, copy=True)
    tmp[tmp == 0] = np.nan
    return float(np.nanmin(tmp))


def reshape_arr(img_arr: np.ndarray) -> Tuple[np.ndarray, list[int]]:
    """
    Reshape by moving the smallest axis to axis=0, then crop empty borders.
    Returns (new_arr, eliminated_coordinates_number=[min0,min1,min2])
    """
    img_shape = np.array(img_arr.shape)
    x, y, z = img_shape[0], img_shape[1], img_shape[2]

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

    new_arr = new_arr[min0 : max0 + 1, min1 : max1 + 1, min2 : max2 + 1]
    return new_arr, [int(min0), int(min1), int(min2)]


def rescale(data: np.ndarray) -> np.ndarray:
    lower_percentile = 1
    upper_percentile = 99
    p_low = np.nanpercentile(data, lower_percentile)
    p_high = np.nanpercentile(data, upper_percentile)

    data_norm = (data - p_low) / (p_high - p_low)
    data_norm = np.clip(data_norm, 0, 1)
    return data_norm


# =========================
# CSV storage helpers
# =========================

def take_csv_file_path(
    output_folder_path: str | Path,
    original_file_path: str | Path,
    original_file_root_path: str | Path,
) -> Path:
    """
    Builds output CSV path preserving cohort/sequence layout:
      output_folder / {cohort}/{sequence}/{case}_MRC.csv
    """
    output_folder_path = Path(output_folder_path)
    original_file_path = Path(original_file_path)
    original_file_root_path = Path(original_file_root_path)

    rel = original_file_path.relative_to(original_file_root_path)
    case_stem = original_file_path.name.replace(".nii.gz", "").replace(".nii", "")
    return output_folder_path / rel.parent / f"{case_stem}_MRC.csv"


def save_results_to_csv(
    data: pd.DataFrame,
    output_folder_path: str | Path,
    original_file_path: str | Path,
    original_file_root_path: str | Path,
) -> Path:
    csv_file_path = take_csv_file_path(output_folder_path, original_file_path, original_file_root_path)
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_file_path.exists():
        old = pd.read_csv(csv_file_path)
        new = pd.concat([old, data], ignore_index=True)
        new.to_csv(csv_file_path, index=False)
    else:
        data.to_csv(csv_file_path, index=False)

    return csv_file_path


def check_if_MRC_completed(
    output_folder_path: str | Path,
    original_file_path: str | Path,
    original_file_root_path: str | Path,
    min_step: int,
    max_step: int,
) -> bool:
    csv_file_path = take_csv_file_path(output_folder_path, original_file_path, original_file_root_path)
    if not csv_file_path.exists():
        return False

    df = pd.read_csv(csv_file_path)
    if df.empty or "step" not in df.columns:
        return False

    return (df["step"].min() <= min_step) and (df["step"].max() >= max_step)


# =========================
# Core class
# =========================

class Feature_Extraction:
    """
    VIC/MRC Feature Extraction class (ported from notebook, minimal dependencies).
    """

    def __init__(
        self,
        stripped: str = "path",
        original: str = "path",
        min_power: int = 0,
        max_power: int = 0,
        power: int = 0,
        min_step: int = 0,
        max_step: int = 0,
        step: int = 0,
        ROI_shape: str = "square",
        primary_rate: Union[str, int] = "all",
        secondary_rate: Union[str, int, list[int]] = "all",
        all_powers: bool = False,
        all_steps: bool = False,
    ):
        self.check_if_file_nifti(stripped)
        self.check_if_file_nifti(original)

        self.stripped_path = stripped
        self.original_path = original
        self.ROI_shape = ROI_shape
        if self.ROI_shape not in {"square", "circle"}:
            raise ValueError(
                f"Unknown ROI_shape: '{self.ROI_shape}'. Supported ROI shapes are 'square' and 'circle'."
            )
        self.primary_rate = primary_rate
        self.secondary_rate = secondary_rate
        self.power = power
        self.secondary_rate_list: list[int] = []

        if isinstance(secondary_rate, list):
            self.secondary_rate_list = secondary_rate
            self.secondary_rate = max(secondary_rate)

        self.sign_list = take_rate_relation_list()

        # sequences
        if all_powers:
            self.power_sq = make_sequence(max_power, min_power)
        else:
            self.power_sq = np.array([power])

        if all_steps:
            self.step_sq = make_sequence(max_step, min_step).tolist()
        else:
            self.step_sq = [step]

        self.columns = ["shape", "primary_rate", "secondary_rate", "step", "power", "MRC_value"]

    def calculate_features(self, calculate_MRC: bool = False):
        self.calculate_MRC = calculate_MRC

        strp_img = nib.load(self.stripped_path)
        org_img = nib.load(self.original_path)

        self.strp_data = strp_img.get_fdata()
        self.org_data = org_img.get_fdata()

        self.data = self.preprocessing_data()
        self.core_denoms = self.take_denominators()

        if self.calculate_MRC:
            self.MRC_results = pd.DataFrame(columns=self.columns)

        for step_size in self.step_sq:
            self.kernel_pix_num = take_pixel_numbers_of_kernel(
                kernel_len=2 * step_size + 1,
                shape=self.ROI_shape,
            )

            self.cluster = self.make_kernel_cluster(step=step_size)
            self.rates: dict[str, np.ndarray] = {}

            self.std_all, self.std_ngb = self.std_calculation()
            self.std_diff = self.std_all - self.std_ngb
            self.std_diff[self.std_diff <= 0] = np.nan

            self.rates["1"] = self.primary_rates_calculation()

            if self.calculate_MRC:
                self.calculate_MRC_results(step_size=step_size, rate_id=1)

            rates_lim = self.find_secondary_rates_limit()
            for j in range(2, rates_lim):
                self.rates[str(j)] = self.rates_calculation(operation_number=j)
                if self.calculate_MRC:
                    self.calculate_MRC_results(step_size=step_size, rate_id=j)

        # if calculate_MRC True, results live in self.MRC_results

    def take_single_feature_list(self) -> list[float]:
        feature_arr = self.rates["1"][0][0].flatten()
        feature_list = feature_arr[np.logical_not(np.isnan(feature_arr))]
        return feature_list.tolist()

    def calculate_MRC_results(self, step_size: int = 1, rate_id: int = 1):
        MRC_df = self.take_MRC_results_df(step_size, rate_id)
        self.MRC_results = concat_2_df(self.MRC_results, MRC_df)

    def preprocessing_data(self) -> np.ndarray:
        data, self.eliminated_coordinates_number = reshape_arr(self.strp_data)
        data = rescale(data)
        data[data == 0] = np.nan
        del self.org_data, self.strp_data
        return data

    def find_secondary_rates_limit(self) -> int:
        if self.secondary_rate == "all":
            return 9
        return int(self.secondary_rate) + 1

    def primary_rates_calculation(self) -> np.ndarray:
        shape = self.data.shape

        if self.primary_rate == "all":
            p_rates = np.zeros((3, len(self.power_sq), shape[0], shape[1], shape[2]))
            for i, p in enumerate(self.power_sq):
                pw_std_all = np.power(self.std_all, p)
                pw_std_diff = np.power(self.std_diff, p)

                p_rates[0][i] = np.divide(pw_std_all, self.std_ngb, out=np.zeros_like(pw_std_all), where=self.std_ngb != 0)
                p_rates[1][i] = np.divide(pw_std_diff, self.std_all, out=np.zeros_like(pw_std_diff), where=self.std_all != 0)
                p_rates[2][i] = np.divide(pw_std_diff, self.std_ngb, out=np.zeros_like(pw_std_diff), where=self.std_ngb != 0)

            del self.std_all, self.std_ngb
            return p_rates

        # single primary rate → 3D
        pw_std_all = np.power(self.std_all, self.power)
        pw_std_diff = np.power(self.std_diff, self.power)

        if int(self.primary_rate) == 1:
            p_rates = np.divide(pw_std_all, self.std_ngb, out=np.zeros_like(pw_std_all), where=self.std_ngb != 0)
        elif int(self.primary_rate) == 2:
            p_rates = np.divide(pw_std_diff, self.std_all, out=np.zeros_like(pw_std_diff), where=self.std_all != 0)
        elif int(self.primary_rate) == 3:
            p_rates = np.divide(pw_std_diff, self.std_ngb, out=np.zeros_like(pw_std_diff), where=self.std_ngb != 0)
        else:
            raise ValueError("primary_rate must be 'all' or 1/2/3")

        del self.std_all, self.std_ngb
        return p_rates

    def take_MRC_results_df(self, step_size: int = 1, rate_id: int = 1) -> pd.DataFrame:
        MRC_info = []

        if self.primary_rate == "all":
            MRC_values = np.nanmax(self.rates[str(rate_id)], axis=(2, 3, 4), initial=0)
            for i, p in enumerate(self.power_sq):
                MRC_info.append([self.ROI_shape, 1, rate_id, step_size, int(p), float(MRC_values[0][i])])
                MRC_info.append([self.ROI_shape, 2, rate_id, step_size, int(p), float(MRC_values[1][i])])
                MRC_info.append([self.ROI_shape, 3, rate_id, step_size, int(p), float(MRC_values[2][i])])
            return pd.DataFrame(MRC_info, columns=self.columns)

        # single primary rate
        MRC_value = float(np.nanmax(self.rates[str(rate_id)], initial=0))
        for p in self.power_sq:
            MRC_info.append([self.ROI_shape, int(self.primary_rate), rate_id, step_size, int(p), MRC_value])
        return pd.DataFrame(MRC_info, columns=self.columns)

    def rates_calculation(self, operation_number: int = 0) -> np.ndarray:
        self.clean_data(operation_number - 1)
        s0, s1, s2 = self.sign_list[operation_number - 2]
        shape = self.data.shape

        if self.primary_rate == "all":
            denom_4 = self.expand_3dim_array_to_4dim(self.core_denoms[s2], len(self.power_sq))
            denom_5 = self.expand_4dim_array_to_5dim(denom_4, 3)

            s_rates = np.divide(self.rates[s1], denom_5, out=np.zeros_like(self.rates[s1]), where=denom_5 != 0)

            # median-centering over spatial axes per (PR, power)
            medians = np.nanmedian(s_rates, axis=(2, 3, 4))  # (3, n_powers)
            expanded = medians[:, :, np.newaxis, np.newaxis, np.newaxis]
            return s_rates - np.broadcast_to(expanded, s_rates.shape)

        # primary_rate != 'all' → 3D
        denom = self.core_denoms[s2]
        s_rates = np.divide(self.rates[s1], denom, out=np.zeros_like(self.rates[s1]), where=denom != 0)

        # ✅ FIX: median-centering for 3D (z,y,x) over spatial axes (y,x) per slice z
        medians = np.nanmedian(s_rates, axis=(1, 2))               # (z,)
        expanded = medians[:, np.newaxis, np.newaxis]              # (z,1,1)
        return s_rates - expanded

    def std_calculation(self) -> Tuple[np.ndarray, np.ndarray]:
        std_all = np.std(self.cluster, axis=1)

        center_index = int((self.kernel_pix_num - 1) / 2)
        self.cluster = np.delete(self.cluster, center_index, axis=1)
        std_ngb = np.std(self.cluster, axis=1)

        del self.cluster
        return std_all, std_ngb

    def make_kernel_cluster(self, step: int = 0) -> np.ndarray:
        data_shape = self.data.shape
        kernel_len = step * 2 + 1

        pad_data = np.zeros((data_shape[0], kernel_len + 1 + data_shape[1], kernel_len + 1 + data_shape[2]))
        pad_data[:, step + 1 : -step - 1, step + 1 : -step - 1] = self.data

        img_shape = self.data.shape
        cluster = np.zeros((img_shape[0], self.kernel_pix_num, img_shape[1], img_shape[2]))

        if self.ROI_shape == "square":
            coordinates_dif = [(x, y) for x in range(-step, step + 1) for y in range(-step, step + 1)]
        elif self.ROI_shape == "circle":
            coordinates_dif = [(x, y) for x in range(-step, step + 1) for y in range(-step, step + 1) if (x**2 + y**2) <= step**2]
        else:
            raise ValueError(f"Unknown ROI_shape: {self.ROI_shape}")

        coordinates_limits = [([step + x + 1, -step + x - 1], [step + y + 1, -step + y - 1]) for x, y in coordinates_dif]
        for i, limit in enumerate(coordinates_limits):
            cluster[:, i, :, :] = pad_data[:, limit[0][0] : limit[0][1], limit[1][0] : limit[1][1]]

        return cluster

    def take_denominators(self) -> dict[str, np.ndarray]:
        denom: dict[str, np.ndarray] = {}

        median_value = np.nanmedian(self.data)
        midrange_value = (np.nanmax(self.data) + np.nanmin(self.data)) / 2

        denom["0"] = np.copy(self.data)

        denom["1"] = np.abs(self.data - median_value)
        denom["1"][denom["1"] == 0] = take_positive_minimum(denom["1"])

        denom["2"] = np.abs(self.data - midrange_value)
        denom["2"][denom["2"] == 0] = take_positive_minimum(denom["2"])

        return denom

    def clean_data(self, operation_number: int):
        if operation_number == 4:
            if 4 not in self.secondary_rate_list:
                self.rates.pop("4", None)
        elif operation_number == 5:
            if 1 not in self.secondary_rate_list:
                self.rates.pop("1", None)
        elif operation_number == 6:
            if 6 not in self.secondary_rate_list:
                self.rates.pop("6", None)
        elif operation_number == 7:
            if 2 not in self.secondary_rate_list:
                self.rates.pop("2", None)
            if 7 not in self.secondary_rate_list:
                self.rates.pop("7", None)
        elif operation_number == 8:
            if 3 not in self.secondary_rate_list:
                self.rates.pop("3", None)

    def expand_3dim_array_to_4dim(self, array: np.ndarray, added_size: int) -> np.ndarray:
        if len(array.shape) != 3:
            raise ValueError("given array is not 3 dimensional")
        array = np.expand_dims(array, axis=0)
        array = np.tile(array, (added_size, 1, 1, 1))
        return array

    def expand_4dim_array_to_5dim(self, array: np.ndarray, added_size: int) -> np.ndarray:
        if len(array.shape) != 4:
            raise ValueError("given array is not 4 dimensional")
        array = np.expand_dims(array, axis=0)
        array = np.tile(array, (added_size, 1, 1, 1, 1))
        return array

    def check_if_file_nifti(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File not found: "{file_path}"')
        if not (file_path.endswith(".nii.gz") or file_path.endswith(".nii")):
            raise ValueError(f'Not a NIfTI file: "{file_path}"')

