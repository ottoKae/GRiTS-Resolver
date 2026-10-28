from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import transform as window_transform

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog
from .provenance import ProvenanceView
from .s2_quality import group_scene_indices
from .spatial import (
    TargetWindow,
    open_dataset,
    read_aligned_variables,
    read_aligned_window,
)
from .temporal import JoinMethod, TemporalMatch, join_times


class MultiSensorResolver:
    def __init__(self, s1_root: str | Path, s2_root: str | Path, *, validate: bool = True,
                 s2_quality_root: str | Path | None = None,
                 s1_metadata_root: str | Path | None = None,
                 s2_metadata_root: str | Path | None = None,
                 require_provenance: bool = False, root_map=None,
                 s2_quality_pairing: str = "legacy"):
        self.s1_root = Path(s1_root).resolve()
        self.s2_root = Path(s2_root).resolve()
        self.provenance = {
            "sentinel-1": ProvenanceView(self.s1_root, metadata_root=s1_metadata_root, required=require_provenance),
            "sentinel-2": ProvenanceView(self.s2_root, metadata_root=s2_metadata_root, required=require_provenance),
        }
        self.quality_companion = None
        self._scene_cache = {}
        self._versioned_s2_stores = {
            (self.s2_root / row["store_path"]).resolve()
            for row in self.provenance["sentinel-2"].snapshot["outputs"]
        }
        self.s1_entries = load_catalog(self.s1_root, "sentinel-1", root_map=root_map)
        self.s2_entries = load_catalog(self.s2_root, "sentinel-2", root_map=root_map)
        if validate:
            self.validate_catalogs()
        if s2_quality_root is not None:
            from .s2_quality import QualityCompanion
            self.quality_companion = QualityCompanion(self.s2_root, s2_quality_root, self.s2_entries,
                                                     pairing=s2_quality_pairing, root_map=root_map)
        elif s2_quality_pairing != "legacy":
            raise ValueError("s2_quality_pairing requires s2_quality_root")
        self._assert_current()

    def _assert_current(self, entries=(), pair=None):
        for sensor, view in getattr(self, "provenance", {}).items():
            view.assert_current(stores=[entry.zarr_path for entry in entries if entry.sensor == sensor])
        companion = getattr(self, "quality_companion", None)
        if companion is not None:
            if pair is None:
                companion.ensure_current()
            else:
                companion.ensure_current(pair)

    @contextmanager
    def _reading(self, *entries, pair=None):
        self._assert_current(entries, pair)
        try:
            yield
        finally:
            self._assert_current(entries, pair)

    def _group_scene_indices(self, entry):
        if entry.zarr_path.resolve() not in getattr(self, "_versioned_s2_stores", set()):
            return group_scene_indices(entry)
        self.provenance["sentinel-2"].assert_current(stores=[entry.zarr_path])
        key = (entry.zarr_path, entry.zarr_group)
        if key not in self._scene_cache:
            self._scene_cache[key] = group_scene_indices(entry)
        return self._scene_cache[key]

    def explain_output(self, sensor: str, **selection):
        """Explain recorded sources and processing without changing stored values."""
        if sensor not in self.provenance:
            raise ValueError("sensor must be sentinel-1 or sentinel-2")
        return self.provenance[sensor].explain(**selection)

    def s2_scene_quality(self, tile_id, item_id, *, collection=None, allow_missing=False):
        self._assert_current()
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
        self._assert_current()
        if self.quality_companion is None:
            raise CatalogConsistencyError("No S2 quality companion configured")
        canonical = self.quality_companion.scene(pair.tile_id, pair.item_id,
                    collection=pair.source_collection, allow_missing=True)
        if pair != canonical:
            raise CatalogConsistencyError("Quality pair differs from this published generation")
        if variable != "scl":
            raise KeyError(variable)
        if target_entry is not None and target_entry.sensor != "sentinel-1":
            raise ValueError("Explicit target must be a Sentinel-1 grid entry")
        if not pair.quality_available:
            with self._reading(pair=pair):
                return {"scl": np.zeros((window.height, window.width), dtype="uint8"),
                        "quality_valid": np.zeros((window.height, window.width), dtype=bool),
                        "quality_available": False, "reason": "QUALITY_MISSING"}
        target = target_entry if target_entry is not None else self.s1_entry(pair.tile_id)
        guarded_targets = (target,) if target_entry is None or target in getattr(self, "s1_entries", ()) else ()
        with self._reading(*guarded_targets, pair=pair):
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
            times, identities = self._group_scene_indices(entries[0])
            zarr_times = np.sort(times)
            keys = [(entry.row.get("source_collection"), entry.row.get("item_id")) for entry in entries]
            if len(set(keys)) != len(keys):
                raise CatalogConsistencyError(f"Duplicate S2 catalog scene identity: {path} group={group}")
            catalog_times = np.sort([
                self._datetime64(entry.row["datetime"]) for entry in entries
            ])
            if not np.array_equal(zarr_times, catalog_times):
                raise CatalogConsistencyError(
                    f"S2 catalog/Zarr time mismatch: {path} group={group}; "
                    f"catalog={len(catalog_times)} zarr={len(zarr_times)}"
                )
            if identities is not None:
                if {entry.row["item_id"] for entry in entries} != set(identities):
                    raise CatalogConsistencyError(f"S2 catalog/commit identities differ: {path} group={group}")
                for entry in entries:
                    index, witness = identities[entry.row["item_id"]]
                    if (times[index] != self._datetime64(entry.row["datetime"]) or
                            witness.get("collection", entry.row.get("source_collection")) != entry.row.get("source_collection") or
                            (witness.get("relative_orbit") is not None and witness["relative_orbit"] != entry.row.get("track"))):
                        raise CatalogConsistencyError(f"S2 catalog/commit identity/UTC differs: {entry.row['item_id']}")

    def entries(self, sensor: str, tile_id: str) -> list[CatalogEntry]:
        self._assert_current()
        source = self.s1_entries if sensor == "sentinel-1" else self.s2_entries
        return [entry for entry in source if entry.tile_id == tile_id]

    @staticmethod
    def _select(entries, *, track=None, store=None, item_id=None, collection=None):
        def track_matches(entry):
            values = (entry.row.get("track"), entry.row.get("relative_orbit"), entry.track_key)
            return any(str(value) == str(track) or
                       (str(value).lstrip("R0") == str(track).lstrip("R0")) for value in values if value is not None)
        return [entry for entry in entries
                if (track is None or track_matches(entry))
                and (store is None or entry.zarr_path.resolve() == Path(store).resolve()
                     or entry.zarr_path.name == str(store))
                and (item_id is None or entry.row.get("item_id") == item_id)
                and (collection is None or entry.row.get("source_collection") == collection)]

    @staticmethod
    def _one_store(matches, label):
        if not matches:
            raise KeyError(f"No {label} entry matches the selection")
        stores = {(entry.zarr_path, entry.zarr_group) for entry in matches}
        if len(stores) != 1:
            raise CatalogConsistencyError(f"Ambiguous {label} stores; select track, item_id or store explicitly")

    def s1_entry(self, tile_id: str, *, track=None, store=None) -> CatalogEntry:
        matches = [
            entry for entry in self.entries("sentinel-1", tile_id)
            if entry.product_type == "smonthly"
        ]
        matches = self._select(matches, track=track, store=store)
        self._one_store(matches, f"S1 {tile_id}")
        return matches[0]

    def s1_tracks(self, tile_id: str) -> list[CatalogEntry]:
        """Return every monthly S1 track; no best-track arbitration is applied."""
        return sorted(
            (entry for entry in self.entries("sentinel-1", tile_id)
             if entry.product_type == "smonthly"),
            key=lambda entry: entry.zarr_path.name,
        )

    def _s2_scene(self, tile_id, resolution_group="20m", *, item_id=None, track=None,
                  store=None, collection=None, time_index=None):
        matches = [
            entry for entry in self.entries("sentinel-2", tile_id)
            if entry.zarr_group == resolution_group
        ]
        matches = self._select(matches, track=track, store=store, item_id=item_id, collection=collection)
        self._one_store(matches, f"S2 {tile_id}/{resolution_group}")
        if time_index is None and item_id is None and track is None:
            tracks = {entry.row.get("track", entry.row.get("relative_orbit")) for entry in matches}
            if len(tracks) > 1:
                raise CatalogConsistencyError("Ambiguous S2 tracks; select time_index, item_id or track explicitly")
        times, identities = self._group_scene_indices(matches[0])
        if time_index is not None and not 0 <= time_index < len(times):
            raise IndexError(f"S2 time_index {time_index} outside committed time axis")
        located = []
        for entry in matches:
            if identities is not None:
                found = identities.get(entry.row.get("item_id"))
                if found is None:
                    raise CatalogConsistencyError("S2 catalog item absent from commit identities")
                index, witness = found
                if (times[index] != self._datetime64(entry.row["datetime"]) or
                        witness.get("collection", entry.row.get("source_collection")) != entry.row.get("source_collection") or
                        (witness.get("relative_orbit") is not None and witness["relative_orbit"] != entry.row.get("track"))):
                    raise CatalogConsistencyError("S2 selected catalog identity/UTC differs from commit")
            elif "datetime" in entry.row:
                indices = np.flatnonzero(times == self._datetime64(entry.row["datetime"]))
                if len(indices) != 1:
                    raise CatalogConsistencyError("S2 UTC is absent or ambiguous; scene commit identities required")
                index = int(indices[0])
            elif len(matches) == 1:
                index = 0 if time_index is None else time_index
            else:
                raise CatalogConsistencyError("S2 scene metadata lacks identity/UTC")
            if time_index is None or index == time_index:
                located.append((index, entry))
        if not located:
            raise CatalogConsistencyError("Selected S2 item/track does not match time_index")
        index, entry = min(located, key=lambda value: value[0])
        if sum(value[0] == index for value in located) != 1:
            raise CatalogConsistencyError("Ambiguous S2 scene identity at time_index")
        return entry, index

    def s2_entry(self, tile_id: str, resolution_group: str = "20m", *, item_id=None,
                 track=None, store=None, collection=None, time_index=None) -> CatalogEntry:
        """Return metadata for the selected committed scene, never another row."""
        return self._s2_scene(tile_id, resolution_group, item_id=item_id, track=track,
                              store=store, collection=collection, time_index=time_index)[0]

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
        s1_track=None,
        s1_store=None,
        s2_store=None,
    ) -> list[TemporalMatch]:
        s1 = self.s1_entry(tile_id, track=s1_track, store=s1_store)
        s2 = self.s2_entry(tile_id, resolution_group, store=s2_store, time_index=0)
        with self._reading(s1, s2):
            ds1, ds2 = open_dataset(s1), open_dataset(s2)
            try:
                t1, t2 = np.asarray(ds1.time.values), np.asarray(ds2.time.values)
            finally:
                ds1.close()
                ds2.close()
            return join_times(t2, t1, method=method, tolerance=tolerance) if left == "s2" else join_times(t1, t2, method=method, tolerance=tolerance)

    def read_s2_on_s1(
        self,
        tile_id: str,
        variable: str,
        *,
        s2_time_index: int | None = None,
        window: TargetWindow,
        resolution_group: str = "20m",
        resampling: Resampling | None = None,
        s1_track=None, s1_store=None,
        s2_item_id=None, s2_track=None, s2_store=None, s2_collection=None,
    ) -> np.ndarray:
        return self.read_s2_variables_on_s1(tile_id, [variable], s2_time_index=s2_time_index,
            window=window, resolution_group=resolution_group,
            resampling={variable: resampling} if resampling is not None else None,
            s1_track=s1_track, s1_store=s1_store, s2_item_id=s2_item_id, s2_track=s2_track,
            s2_store=s2_store, s2_collection=s2_collection)[variable]

    def read_s2_variables_on_s1(
        self,
        tile_id: str,
        variables: list[str],
        *,
        s2_time_index: int | None = None,
        window: TargetWindow,
        resolution_group: str = "20m",
        resampling: dict[str, Resampling] | None = None,
        s1_track=None, s1_store=None,
        s2_item_id=None, s2_track=None, s2_store=None, s2_collection=None,
    ) -> dict[str, np.ndarray]:
        if s2_time_index is None and s2_item_id is None:
            raise ValueError("Select s2_item_id or s2_time_index explicitly")
        source, index = self._s2_scene(tile_id, resolution_group, item_id=s2_item_id,
            track=s2_track, store=s2_store, collection=s2_collection, time_index=s2_time_index)
        target = self.s1_entry(tile_id, track=s1_track, store=s1_store)
        with self._reading(source, target):
            pair, coverage = None, None
            if {"scl", "cloud_probability"}.intersection(variables) and self.quality_companion is not None:
                pair = self.quality_companion.scene(tile_id, source.row["item_id"],
                                                   collection=source.row["source_collection"])
                coverage = (pair.quality_20m, pair.quality_20m_index)
            if "scl" in variables and self.quality_companion is not None:
                if (resampling or {}).get("scl", Resampling.nearest) != Resampling.nearest:
                    raise ValueError("SCL requires nearest resampling")
                ordinary = [name for name in variables if name != "scl"]
                result = read_aligned_variables(source, target, ordinary, time_index=index,
                    window=window, resampling=resampling, quality_coverage=coverage) if ordinary else {}
                result["scl"] = self.read_s2_quality_on_s1(pair, window=window, target_entry=target)["scl"]
                return result
            return read_aligned_variables(
                source, target, variables, time_index=index, window=window, resampling=resampling,
                quality_coverage=coverage,
            )
