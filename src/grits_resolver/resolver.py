from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.enums import Resampling

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog
from .spatial import TargetWindow, open_dataset, read_aligned_variables, read_aligned_window
from .temporal import JoinMethod, TemporalMatch, join_times


class MultiSensorResolver:
    def __init__(self, s1_root: str | Path, s2_root: str | Path, *, validate: bool = True):
        self.s1_root = Path(s1_root).resolve()
        self.s2_root = Path(s2_root).resolve()
        self.s1_entries = load_catalog(self.s1_root, "sentinel-1")
        self.s2_entries = load_catalog(self.s2_root, "sentinel-2")
        if validate:
            self.validate_catalogs()

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
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
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
        return read_aligned_variables(
            self.s2_entry(tile_id, resolution_group), self.s1_entry(tile_id), variables,
            time_index=s2_time_index, window=window, resampling=resampling,
        )
