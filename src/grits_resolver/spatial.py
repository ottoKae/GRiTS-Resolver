from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, transform as window_transform

from .catalog import CatalogEntry


NEAREST_VARIABLES = {"scl", "cloud_probability", "mask", "cloud_mask", "quality_mask"}


@dataclass(frozen=True)
class TargetWindow:
    row_off: int
    col_off: int
    height: int
    width: int

    def rasterio_window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def open_dataset(entry: CatalogEntry, *, raw: bool = False) -> xr.Dataset:
    options = {"mask_and_scale": False, "use_zarr_fill_value_as_mask": False} if raw else {}
    return xr.open_zarr(entry.zarr_path, group=entry.zarr_group, consolidated=False,
                        chunks=None, **options)


def resampling_for_variable(variable: str) -> Resampling:
    name = variable.lower()
    return Resampling.nearest if name in NEAREST_VARIABLES or name.endswith("_mask") else Resampling.bilinear


def _windows(source: CatalogEntry, target: CatalogEntry, window: TargetWindow):
    target_affine = Affine(*target.transform)
    dst_transform = window_transform(window.rasterio_window(), target_affine)
    dst_bounds = array_bounds(window.height, window.width, dst_transform)
    left, bottom, right, top = transform_bounds(target.crs, source.crs, *dst_bounds)
    source_affine = Affine(*source.transform)
    inverse = ~source_affine
    pixels = [inverse * (x, y) for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))]
    col0 = max(0, math.floor(min(pixel[0] for pixel in pixels)))
    col1 = min(source.width, math.ceil(max(pixel[0] for pixel in pixels)))
    row0 = max(0, math.floor(min(pixel[1] for pixel in pixels)))
    row1 = min(source.height, math.ceil(max(pixel[1] for pixel in pixels)))
    if col1 <= col0 or row1 <= row0:
        return None, dst_transform
    return Window(col0, row0, col1 - col0, row1 - row0), dst_transform


def read_aligned_variables(
    source_entry: CatalogEntry,
    target_entry: CatalogEntry,
    variables: list[str],
    *,
    time_index: int,
    window: TargetWindow,
    resampling: dict[str, Resampling] | None = None,
) -> dict[str, np.ndarray]:
    """Open one Zarr group once and align several variables for one target window."""
    if "scl" in variables and (resampling or {}).get("scl", Resampling.nearest) != Resampling.nearest:
        raise ValueError("SCL is categorical and requires nearest resampling")
    src_window, dst_transform = _windows(source_entry, target_entry, window)
    if src_window is None:
        return {name: (np.zeros((window.height, window.width), dtype="uint8") if name == "scl"
                       else np.full((window.height, window.width), np.nan, dtype="float32")) for name in variables}
    row0, col0 = int(src_window.row_off), int(src_window.col_off)
    row1, col1 = row0 + int(src_window.height), col0 + int(src_window.width)
    source_affine = Affine(*source_entry.transform)
    dataset = open_dataset(source_entry, raw="scl" in variables)
    try:
        missing = [name for name in variables if name not in dataset]
        if missing:
            raise KeyError(f"Variables not found in {source_entry.zarr_group}: {missing}")
        outputs: dict[str, np.ndarray] = {}
        for variable in variables:
            array = dataset[variable]
            source = np.asarray(
                array.isel(time=time_index, y=slice(row0, row1), x=slice(col0, col1))
            )
            method = (resampling or {}).get(variable, resampling_for_variable(variable))
            nodata = array.attrs.get("nodata", array.attrs.get("_FillValue", 0))
            categorical = method == Resampling.nearest
            destination = (
                np.full((window.height, window.width), nodata, dtype=source.dtype)
                if categorical
                else np.full((window.height, window.width), np.nan, dtype="float32")
            )
            reproject(
                source=source,
                destination=destination,
                src_transform=window_transform(src_window, source_affine),
                src_crs=source_entry.crs,
                dst_transform=dst_transform,
                dst_crs=target_entry.crs,
                src_nodata=nodata,
                dst_nodata=nodata if categorical else np.nan,
                resampling=method,
            )
            outputs[variable] = destination
        return outputs
    finally:
        dataset.close()


def read_aligned_window(
    source_entry: CatalogEntry,
    target_entry: CatalogEntry,
    variable: str,
    *,
    time_index: int,
    window: TargetWindow,
    resampling: Resampling | None = None,
) -> np.ndarray:
    policies = {variable: resampling} if resampling is not None else None
    return read_aligned_variables(
        source_entry, target_entry, [variable], time_index=time_index,
        window=window, resampling=policies,
    )[variable]
