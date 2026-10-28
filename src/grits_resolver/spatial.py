from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
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


def _warp_scale(source: CatalogEntry, target: CatalogEntry):
    # shortcut: fixed scale suits local MGRS/UTM grids; use adaptive kernels for large nonlinear reprojections.
    bounds = transform_bounds(target.crs, source.crs,
                              *array_bounds(target.height, target.width, Affine(*target.transform)))
    inverse = ~Affine(*source.transform)
    pixels = [inverse * (x, y) for x, y in ((bounds[0], bounds[1]), (bounds[0], bounds[3]),
                                            (bounds[2], bounds[1]), (bounds[2], bounds[3]))]
    return (max(1., (max(p[0] for p in pixels) - min(p[0] for p in pixels)) / target.width),
            max(1., (max(p[1] for p in pixels) - min(p[1] for p in pixels)) / target.height))


def _windows(source: CatalogEntry, target: CatalogEntry, window: TargetWindow, halo=(0, 0)):
    target_affine = Affine(*target.transform)
    dst_transform = window_transform(window.rasterio_window(), target_affine)
    dst_bounds = array_bounds(window.height, window.width, dst_transform)
    left, bottom, right, top = transform_bounds(target.crs, source.crs, *dst_bounds)
    source_affine = Affine(*source.transform)
    inverse = ~source_affine
    pixels = [inverse * (x, y) for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))]
    col0 = max(0, math.floor(min(pixel[0] for pixel in pixels)) - halo[0])
    col1 = min(source.width, math.ceil(max(pixel[0] for pixel in pixels)) + halo[0])
    row0 = max(0, math.floor(min(pixel[1] for pixel in pixels)) - halo[1])
    row1 = min(source.height, math.ceil(max(pixel[1] for pixel in pixels)) + halo[1])
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
    quality_coverage: tuple[CatalogEntry, int] | None = None,
) -> dict[str, np.ndarray]:
    """Align raw DN consistently; scale/offset remain the caller's responsibility.

    Cloud probability is 0..100 with 255 outside same-scene SCL coverage. SCL
    remains uint8 with zero outside coverage; other masks keep explicit nodata.
    """
    if window.height < 1 or window.width < 1 or time_index < 0:
        raise ValueError("Expected a positive window and nonnegative time_index")
    if "scl" in variables and (resampling or {}).get("scl", Resampling.nearest) != Resampling.nearest:
        raise ValueError("SCL is categorical and requires nearest resampling")
    methods = {name: (resampling or {}).get(name, resampling_for_variable(name)) for name in variables}
    scale = _warp_scale(source_entry, target_entry)
    radius = max((3 if method == Resampling.lanczos else 2 if method in
                  {Resampling.cubic, Resampling.cubic_spline} else 1
                  for method in methods.values()), default=1)
    halo = tuple(math.ceil(radius * value) + 2 for value in scale)
    src_window, dst_transform = _windows(source_entry, target_entry, window, halo)
    if src_window is None:
        return {name: (np.full((window.height, window.width), 255, dtype="uint8") if name == "cloud_probability"
                       else np.zeros((window.height, window.width), dtype="uint8") if name == "scl"
                       else np.full((window.height, window.width), np.nan, dtype="float32")) for name in variables}
    row0, col0 = int(src_window.row_off), int(src_window.col_off)
    row1, col1 = row0 + int(src_window.height), col0 + int(src_window.width)
    source_affine = Affine(*source_entry.transform)
    dataset = open_dataset(source_entry, raw=True)
    quality = None
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
            method = methods[variable]
            nodata = array.attrs.get("nodata", array.attrs.get("_FillValue"))
            if variable == "scl":
                nodata = 0
            elif variable == "cloud_probability":
                coverage_dataset, coverage_index = dataset, time_index
                if "scl" not in dataset and quality_coverage is not None:
                    quality_entry, coverage_index = quality_coverage
                    if any(getattr(quality_entry, field) != getattr(source_entry, field)
                           for field in ("crs", "transform", "width", "height")):
                        raise ValueError("Cloud probability/SCL coverage grids differ")
                    quality = open_dataset(quality_entry, raw=True)
                    coverage_dataset = quality
                if "scl" not in coverage_dataset:
                    raise ValueError("Cloud probability requires same-scene SCL coverage to preserve valid zero")
                scl = np.asarray(coverage_dataset.scl.isel(time=coverage_index, y=slice(row0, row1), x=slice(col0, col1)))
                valid = (scl > 0) & (scl <= 11) & np.isfinite(source) & (source >= 0) & (source <= 100)
                source = np.where(valid, source, 255).astype("uint8")
                nodata = 255
            categorical = method == Resampling.nearest
            output_dtype = source.dtype.name if categorical and nodata is not None else "float32"
            # Fixed grid-derived kernel scale and tight transform tolerance keep
            # point, block and whole-window reads numerically consistent.
            with MemoryFile() as memory:
                with memory.open(driver="GTiff", width=source.shape[1], height=source.shape[0],
                                 count=1, dtype=source.dtype, crs=source_entry.crs, nodata=nodata,
                                 transform=window_transform(src_window, source_affine)) as temporary:
                    temporary.write(source, 1)
                with memory.open() as temporary:
                    with WarpedVRT(temporary, crs=target_entry.crs, transform=dst_transform,
                                   width=window.width, height=window.height,
                                   dtype=output_dtype, nodata=nodata if categorical and nodata is not None else np.nan,
                                   resampling=method, tolerance=1e-9,
                                   XSCALE=str(1 / scale[0]), YSCALE=str(1 / scale[1])) as warped:
                        destination = warped.read(1)
            outputs[variable] = destination
        return outputs
    finally:
        dataset.close()
        if quality is not None:
            quality.close()


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
