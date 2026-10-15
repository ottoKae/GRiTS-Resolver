from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import transform as window_transform

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog
from .spatial import TargetWindow, open_dataset, read_aligned_variables, read_aligned_window
from .temporal import JoinMethod, TemporalMatch, join_times


class MultiSensorResolver:
    def __init__(self, s1_root: str | Path, s2_root: str | Path, *, validate: bool = True,
                 s2_quality_root: str | Path | None = None):
        self.s1_root = Path(s1_root).resolve()
        self.s2_root = Path(s2_root).resolve()
        self.s1_entries = load_catalog(self.s1_root, "sentinel-1")
        self.s2_entries = load_catalog(self.s2_root, "sentinel-2")
        if validate:
            self.validate_catalogs()
        self.quality_companion = None
        if s2_quality_root is not None:
            from .s2_quality import QualityCompanion
            self.quality_companion = QualityCompanion(self.s2_root, s2_quality_root, self.s2_entries)

    def s2_scene_quality(self, tile_id, item_id, *, collection=None, allow_missing=False):
        if self.quality_companion is None:
            raise CatalogConsistencyError("No S2 quality companion configured")
        return self.quality_companion.scene(tile_id, item_id, collection=collection, allow_missing=allow_missing)

    def s2_quality_candidates_on_target(self, target_entry, *, window, start, end,
                                        allow_missing=True):
        """Return stable scene pairs intersecting a target window, across tiles.

        Dates use [start, end). Raster bounds deliberately give a conservative
        candidate set, not a guarantee of scene footprint, clear or valid data.
        This does not mosaic, deduplicate acquisitions or change source weights.
        """
        if self.quality_companion is None:
            raise CatalogConsistencyError("No S2 quality companion configured")
        if target_entry.sensor != "sentinel-1":
            raise ValueError("Explicit target must be a Sentinel-1 grid entry")
        begin, stop = self._datetime64(start), self._datetime64(end)
        if np.isnat(begin) or np.isnat(stop) or begin >= stop:
            raise ValueError("Expected finite start < end (exclusive)")
        self.quality_companion.ensure_current()
        tf = window_transform(window.rasterio_window(), Affine(*target_entry.transform))
        bounds = array_bounds(window.height, window.width, tf)
        projected = {}
        selected = {}
        for source in self.s2_entries:
            if source.zarr_group != "10m":
                continue
            when = self._datetime64(source.row["datetime"])
            if not begin <= when < stop:
                continue
            if source.crs not in projected:
                projected[source.crs] = transform_bounds(target_entry.crs, source.crs, *bounds)
            left, bottom, right, top = projected[source.crs]
            sl, sb, sr, st = array_bounds(source.height, source.width, Affine(*source.transform))
            if min(right, sr) <= max(left, sl) or min(top, st) <= max(bottom, sb):
                continue
            identity = (str(source.row["source_collection"]), source.tile_id, str(source.row["item_id"]))
            selected[identity] = when
        return [self.quality_companion.scene(tile, item, collection=collection,
                    allow_missing=allow_missing)
                for (collection, tile, item), _ in sorted(selected.items(), key=lambda v: (v[1], v[0]))]

    def read_s2_quality_on_s1(self, pair, *, variable="scl", window,
                             target_entry: CatalogEntry | None = None):
        """Read categorical SCL onto an explicit target, or legacy same-tile grid.

        Nonzero quality_valid means source coverage, not clear-sky acceptance.
        Nearest output does not implement full-footprint cloud screening.
        """
        if self.quality_companion is None:
            raise CatalogConsistencyError("No S2 quality companion configured")
        self.quality_companion.ensure_current()
        canonical = self.quality_companion.scene(pair.tile_id, pair.item_id,
                    collection=pair.source_collection, allow_missing=True)
        if pair != canonical:
            raise CatalogConsistencyError("Quality pair differs from this published generation")
        if variable != "scl":
            raise KeyError(variable)
        if target_entry is not None and target_entry.sensor != "sentinel-1":
            raise ValueError("Explicit target must be a Sentinel-1 grid entry")
        if not pair.quality_available:
            return {"scl": np.zeros((window.height, window.width), dtype="uint8"),
                    "quality_valid": np.zeros((window.height, window.width), dtype=bool),
                    "quality_available": False, "reason": "QUALITY_MISSING"}
        target = target_entry if target_entry is not None else self.s1_entry(pair.tile_id)
        result = read_aligned_window(pair.quality_20m, target, "scl",
            time_index=pair.quality_20m_index, window=window, resampling=Resampling.nearest)
        if result.dtype != np.dtype("uint8"):
            raise CatalogConsistencyError("SCL raw dtype is not uint8")
        return {"scl": result, "quality_valid": result != 0,
                "quality_available": True, "reason": "AVAILABLE"}

    @staticmethod
    def _datetime64(value) -> np.datetime64:
        return np.datetime64(str(value).replace("+00:00", "").replace("Z", ""), "ns")

    def validate_catalogs(self) -> None:
        """Require catalog rows and Zarr time axes to describe the same commits."""
        for entry in self.s1_entries:
            if not entry.zarr_path.exists():
                raise CatalogConsistencyError(f"Missing S1 Zarr: {entry.zarr_path}")
            if entry.product_type != "smonthly":
                continue
            dataset = open_dataset(entry)
            try:
                times = np.asarray(dataset.time.values).astype("datetime64[ns]")
            finally:
                dataset.close()
            if times.size == 0:
                raise CatalogConsistencyError(f"Empty S1 time axis: {entry.zarr_path}")
            start, end = entry.row.get("start_datetime"), entry.row.get("end_datetime")
            if start is not None and times.min() != self._datetime64(start):
                raise CatalogConsistencyError(f"S1 catalog/Zarr start time mismatch: {entry.zarr_path}")
            if end is not None and times.max() != self._datetime64(end):
                raise CatalogConsistencyError(f"S1 catalog/Zarr end time mismatch: {entry.zarr_path}")

        groups: dict[tuple[Path, str | None], list[CatalogEntry]] = {}
        for entry in self.s2_entries:
            groups.setdefault((entry.zarr_path, entry.zarr_group), []).append(entry)
        for (path, group), entries in groups.items():
            if not path.exists():
                raise CatalogConsistencyError(f"Missing S2 Zarr: {path}")
            dataset = open_dataset(entries[0])
            try:
                zarr_times = np.sort(np.asarray(dataset.time.values).astype("datetime64[ns]"))
            finally:
                dataset.close()
            catalog_times = np.sort(np.unique([
                self._datetime64(entry.row["datetime"]) for entry in entries
            ]))
            if not np.array_equal(zarr_times, catalog_times):
                raise CatalogConsistencyError(
                    f"S2 catalog/Zarr time mismatch: {path} group={group}; "
                    f"catalog={len(catalog_times)} zarr={len(zarr_times)}"
                )

    def entries(self, sensor: str, tile_id: str) -> list[CatalogEntry]:
        source = self.s1_entries if sensor == "sentinel-1" else self.s2_entries
        return [entry for entry in source if entry.tile_id == tile_id]

    def s1_entry(self, tile_id: str) -> CatalogEntry:
        matches = [
            entry for entry in self.entries("sentinel-1", tile_id)
            if entry.product_type == "smonthly"
        ]
        if not matches:
            raise KeyError(f"No S1 entry for {tile_id}")
        return matches[0]

    def s1_tracks(self, tile_id: str) -> list[CatalogEntry]:
        """Return every monthly S1 track; no best-track arbitration is applied."""
        return sorted(
            (entry for entry in self.entries("sentinel-1", tile_id)
             if entry.product_type == "smonthly"),
            key=lambda entry: entry.zarr_path.name,
        )

    def s2_entry(self, tile_id: str, resolution_group: str = "20m") -> CatalogEntry:
        matches = [
            entry for entry in self.entries("sentinel-2", tile_id)
            if entry.zarr_group == resolution_group
        ]
        if not matches:
            raise KeyError(f"No S2 {resolution_group} entry for {tile_id}")
        return matches[0]

    @staticmethod
    def spatial_alignment_mode(source: CatalogEntry, target: CatalogEntry) -> str:
        return "exact" if source.grid_id == target.grid_id else "virtual_reprojection"

    def time_join(
        self,
        tile_id: str,
        *,
        resolution_group: str = "20m",
        method: JoinMethod = "nearest",
        tolerance: np.timedelta64 | None = None,
        left: str = "s2",
    ) -> list[TemporalMatch]:
        s1, s2 = self.s1_entry(tile_id), self.s2_entry(tile_id, resolution_group)
        ds1, ds2 = open_dataset(s1), open_dataset(s2)
        try:
            t1, t2 = np.asarray(ds1.time.values), np.asarray(ds2.time.values)
        finally:
            ds1.close(); ds2.close()
        return join_times(t2, t1, method=method, tolerance=tolerance) if left == "s2" else join_times(t1, t2, method=method, tolerance=tolerance)

    def read_s2_on_s1(
        self,
        tile_id: str,
        variable: str,
        *,
        s2_time_index: int,
        window: TargetWindow,
        resolution_group: str = "20m",
        resampling: Resampling | None = None,
    ) -> np.ndarray:
        if variable == "scl":
            if resampling is not None and resampling != Resampling.nearest:
                raise ValueError("SCL requires nearest resampling")
            if self.quality_companion is not None:
                pair = self.quality_companion.scene_at(tile_id, resolution_group, s2_time_index)
                return self.read_s2_quality_on_s1(pair, window=window)["scl"]
        return read_aligned_window(
            self.s2_entry(tile_id, resolution_group), self.s1_entry(tile_id), variable,
            time_index=s2_time_index, window=window, resampling=resampling,
        )

    def read_s2_variables_on_s1(
        self,
        tile_id: str,
        variables: list[str],
        *,
        s2_time_index: int,
        window: TargetWindow,
        resolution_group: str = "20m",
        resampling: dict[str, Resampling] | None = None,
    ) -> dict[str, np.ndarray]:
        if "scl" in variables and self.quality_companion is not None:
            if (resampling or {}).get("scl", Resampling.nearest) != Resampling.nearest:
                raise ValueError("SCL requires nearest resampling")
            pair = self.quality_companion.scene_at(tile_id, resolution_group, s2_time_index)
            ordinary = [name for name in variables if name != "scl"]
            result = self.read_s2_variables_on_s1(tile_id, ordinary, s2_time_index=s2_time_index,
                window=window, resolution_group=resolution_group, resampling=resampling) if ordinary else {}
            result["scl"] = self.read_s2_quality_on_s1(pair, window=window)["scl"]
            return result
        return read_aligned_variables(
            self.s2_entry(tile_id, resolution_group), self.s1_entry(tile_id), variables,
            time_index=s2_time_index, window=window, resampling=resampling,
        )
