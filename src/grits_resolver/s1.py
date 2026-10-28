from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog
from .provenance import ProvenanceView
from .spatial import TargetWindow, open_dataset

DEFAULT_BANDS = ("VV_dB", "VH_dB")
STATIC_BANDS = (
    "local_inc_angle", "inc_angle", "ls_map", "number_of_looks",
    "rtc_anf_beta0", "rtc_anf_sigma0",
)


@dataclass(frozen=True)
class S1TrackPair:
    dynamic: CatalogEntry
    static: CatalogEntry | None
    static_match: str | None = None
    provenance: ProvenanceView | None = field(default=None, repr=False, compare=False)

    @property
    def monthly(self) -> CatalogEntry:
        """Backward-compatible name for callers resolving smonthly products."""
        return self.dynamic

    @property
    def track_key(self) -> str:
        return self.dynamic.track_key


def _reference_path(entry: CatalogEntry) -> str | None:
    value = entry.row.get("reference_zarr_path")
    if not value:
        return None
    return Path(str(value)).as_posix().lower()


class S1GRiTSResolver:
    """Track-preserving resolver for monthly S1 and matching RTC-STATIC stores.

    No arbitration is performed by this class. A tile with two catalogued
    monthly tracks returns two :class:`S1TrackPair` objects. This is the
    required contract for same-coordinate cross-track diagnostics.
    """

    def __init__(self, dataset_root: str | Path, *, validate: bool = True,
                 metadata_root: str | Path | None = None,
                 require_provenance: bool = False, root_map=None):
        self.root = Path(dataset_root).resolve()
        self.provenance = ProvenanceView(self.root, metadata_root=metadata_root, required=require_provenance)
        self.entries = load_catalog(self.root, "sentinel-1", root_map=root_map)
        self.monthly = self._unique([e for e in self.entries if e.product_type == "smonthly"])
        self.scenes = self._unique([e for e in self.entries if e.product_type == "scenes"])
        self.static = self._unique([e for e in self.entries if e.product_type == "static"])
        if validate:
            self.validate()
        self.provenance.assert_current()

    def explain_output(self, **selection):
        """Return recorded evidence, including explicit unknown historical fields."""
        return self.provenance.explain(**selection)

    def validate(self) -> None:
        for entry in self.entries:
            if not entry.zarr_path.exists():
                raise CatalogConsistencyError(f"Missing S1 Zarr: {entry.zarr_path}")
        for dynamic in [*self.monthly, *self.scenes]:
            dataset = open_dataset(dynamic)
            try:
                if "time" not in dataset.coords or dataset.sizes.get("time", 0) == 0:
                    raise CatalogConsistencyError(f"Empty dynamic time axis: {dynamic.zarr_path}")
            finally:
                dataset.close()
        # A static product must name the exact monthly store it describes.
        for static in self.static:
            if not _reference_path(static):
                raise CatalogConsistencyError(
                    f"Static product lacks reference_zarr_path: {static.zarr_path}"
                )

    def tiles(self) -> list[str]:
        return sorted({entry.tile_id for entry in [*self.monthly, *self.scenes, *self.static]})

    @staticmethod
    def _unique(entries: list[CatalogEntry]) -> list[CatalogEntry]:
        """Collapse acquisition-level catalog rows to unique Zarr products."""
        unique: dict[tuple[Path, str | None], CatalogEntry] = {}
        for entry in entries:
            key = (entry.zarr_path, entry.zarr_group)
            previous = unique.get(key)
            if previous is not None:
                fields = ("tile_id", "product_type", "grid_id", "crs", "transform", "width", "height")
                if any(getattr(previous, field) != getattr(entry, field) for field in fields):
                    raise CatalogConsistencyError(f"Conflicting catalog rows for {entry.zarr_path}")
            else:
                unique[key] = entry
        return sorted(unique.values(), key=lambda entry: entry.zarr_path.name)

    def product_tracks(self, tile_id: str, product_type: str) -> list[CatalogEntry]:
        """Return every unique per-track store for scenes, smonthly, or static."""
        self.provenance.assert_current()
        sources = {"scenes": self.scenes, "smonthly": self.monthly, "static": self.static}
        if product_type not in sources:
            raise ValueError("product_type must be scenes, smonthly, or static")
        return [entry for entry in sources[product_type] if entry.tile_id == str(tile_id)]

    def static_tracks(self, tile_id: str) -> list[CatalogEntry]:
        return self.product_tracks(tile_id, "static")

    def tracks(self, tile_id: str, *, product_type: str = "smonthly",
               require_static: bool = False,
               strict_static_reference: bool = False) -> list[S1TrackPair]:
        """Return all dynamic tracks and their geometry-correct static companion."""
        if product_type not in {"smonthly", "scenes"}:
            raise ValueError("dynamic product_type must be smonthly or scenes")
        dynamic_entries = self.product_tracks(tile_id, product_type)
        pairs: list[S1TrackPair] = []
        for entry in dynamic_entries:
            relative_candidates = {
                entry.zarr_path.as_posix().lower(),
                Path(str(entry.row.get("zarr_path"))).as_posix().lower(),
            }
            exact = [
                static for static in self.static
                if static.tile_id == entry.tile_id
                and _reference_path(static) in relative_candidates
            ]
            geometric = [
                static for static in self.static
                if static.tile_id == entry.tile_id
                and static.track_key == entry.track_key
                and static.grid_id == entry.grid_id
            ]
            matches = exact or geometric
            if strict_static_reference and not exact:
                if geometric:
                    raise CatalogConsistencyError(
                        f"RTC-STATIC pair is only a geometry-group fallback for "
                        f"{entry.zarr_path}; an exact reference_zarr_path is required"
                    )
                raise CatalogConsistencyError(
                    f"No exact RTC-STATIC reference for {entry.zarr_path}"
                )
            if len(matches) > 1:
                raise CatalogConsistencyError(
                    f"Multiple static stores reference monthly track {entry.zarr_path}: "
                    f"{[str(e.zarr_path) for e in matches]}"
                )
            static = matches[0] if matches else None
            if require_static and static is None:
                raise CatalogConsistencyError(f"No RTC-STATIC pair for {entry.zarr_path}")
            if static is not None:
                fields = ("grid_id", "crs", "transform", "width", "height")
                bad = [name for name in fields if getattr(entry, name) != getattr(static, name)]
                if bad:
                    raise CatalogConsistencyError(
                        f"Monthly/static grid mismatch for {entry.zarr_path}: {bad}"
                    )
            pairs.append(S1TrackPair(entry, static, "reference" if exact else "geometry_group", self.provenance))
        return pairs

    @staticmethod
    def pixel_for_coordinate(entry: CatalogEntry, x: float, y: float) -> tuple[int, int]:
        """Convert a coordinate already expressed in the entry CRS to row/column."""
        col_f, row_f = (~Affine(*entry.transform)) * (float(x), float(y))
        row, col = int(np.floor(row_f)), int(np.floor(col_f))
        if not (0 <= row < entry.height and 0 <= col < entry.width):
            raise ValueError(f"Coordinate {(x, y)} is outside {entry.tile_id}")
        return row, col

    @staticmethod
    def _window(row: int, col: int, patch_size: int, entry: CatalogEntry) -> TargetWindow:
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        radius = patch_size // 2
        window = TargetWindow(row - radius, col - radius, patch_size, patch_size)
        if (window.row_off < 0 or window.col_off < 0
                or window.row_off + window.height > entry.height
                or window.col_off + window.width > entry.width):
            raise ValueError("Patch crosses the catalog grid boundary")
        return window

    @staticmethod
    def read_monthly(pair: S1TrackPair, row: int, col: int, *, patch_size: int = 9,
                     bands=DEFAULT_BANDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read one track without arbitration as values, valid mask and month labels."""
        window = S1GRiTSResolver._window(row, col, patch_size, pair.monthly)
        if pair.provenance is not None:
            pair.provenance.assert_current(stores=[pair.monthly.zarr_path])
        ds = open_dataset(pair.monthly)
        try:
            arrays = np.stack([
                np.asarray(ds[name].isel(
                    y=slice(window.row_off, window.row_off + window.height),
                    x=slice(window.col_off, window.col_off + window.width),
                ).values, dtype="float32") for name in bands
            ])
            valid = np.isfinite(arrays).all(axis=0)
            if "n_obs" in ds:
                n_obs = np.asarray(ds["n_obs"].isel(
                    y=slice(window.row_off, window.row_off + window.height),
                    x=slice(window.col_off, window.col_off + window.width),
                ).values)
                valid &= n_obs > 0
            arrays = np.where(valid[None], arrays, 0.0).astype("float32", copy=False)
            months = pd.DatetimeIndex(ds.time.values).to_period("M").astype(str).to_numpy()
            return arrays, valid, months
        finally:
            ds.close()
            if pair.provenance is not None:
                pair.provenance.assert_current(stores=[pair.monthly.zarr_path])

    @staticmethod
    def read_static(pair: S1TrackPair, row: int, col: int, *, patch_size: int = 9,
                    bands=STATIC_BANDS) -> tuple[np.ndarray, np.ndarray]:
        if pair.static is None:
            raise CatalogConsistencyError(f"No RTC-STATIC pair for {pair.monthly.zarr_path}")
        window = S1GRiTSResolver._window(row, col, patch_size, pair.static)
        if pair.provenance is not None:
            pair.provenance.assert_current(stores=[pair.static.zarr_path])
        ds = open_dataset(pair.static)
        try:
            arrays = np.stack([
                np.asarray(ds[name].isel(
                    y=slice(window.row_off, window.row_off + window.height),
                    x=slice(window.col_off, window.col_off + window.width),
                ).values, dtype="float32") for name in bands
            ])
            return arrays, np.isfinite(arrays).all(axis=0)
        finally:
            ds.close()
            if pair.provenance is not None:
                pair.provenance.assert_current(stores=[pair.static.zarr_path])

    @staticmethod
    def common_months(first_months: np.ndarray, second_months: np.ndarray) -> np.ndarray:
        """Sorted calendar intersection used to remove fixed track gaps."""
        return np.intersect1d(np.asarray(first_months, dtype=str), np.asarray(second_months, dtype=str))

    def read_common_pair(self, tile_id: str, row: int, col: int, *, patch_size: int = 9,
                         bands=DEFAULT_BANDS) -> dict[str, object]:
        """Read the two tracks on one calendar and force an identical valid mask.

        Calendar intersection removes store-level fixed gaps. The subsequent
        logical AND removes pixel/month observations that exist in only one
        track. Both returned arrays are zero outside ``common_valid``.
        """
        pairs = self.tracks(tile_id, product_type="smonthly", require_static=True)
        if len(pairs) != 2:
            raise ValueError(f"{tile_id} has {len(pairs)} monthly tracks; exactly two required")
        stores = [entry.zarr_path for pair in pairs for entry in (pair.monthly, pair.static) if entry is not None]
        self.provenance.assert_current(stores=stores)
        try:
            reads = [self.read_monthly(pair, row, col, patch_size=patch_size, bands=bands)
                     for pair in pairs]
            common = self.common_months(reads[0][2], reads[1][2])
            indices = [np.asarray([np.flatnonzero(months == month)[0] for month in common], dtype=np.intp)
                       for _, _, months in reads]
            values = [read[0][:, index] for read, index in zip(reads, indices)]
            valid = [read[1][index] for read, index in zip(reads, indices)]
            common_valid = valid[0] & valid[1]
            values = [np.where(common_valid[None], value, 0.0).astype("float32", copy=False)
                      for value in values]
            static = [self.read_static(pair, row, col, patch_size=patch_size)
                      for pair in pairs]
            return {
                "tracks": pairs,
                "months": common,
                "sar": values,
                "common_valid": common_valid,
                "static": [item[0] for item in static],
                "static_valid": [item[1] for item in static],
            }
        finally:
            self.provenance.assert_current(stores=stores)
