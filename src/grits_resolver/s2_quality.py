"""Read-only, generation-bound S2 spectral/SCL scene pairing."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog


def require(condition, message):
    if not condition:
        raise CatalogConsistencyError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stamp(value):
    return np.datetime64(str(value).replace("Z", "").replace("+00:00", ""), "ns")


def key(entry):
    return (str(entry.row["source_collection"]), entry.tile_id, str(entry.row["item_id"]))


def child(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute(), "Companion paths must be root-relative")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and path.resolve() != root.resolve(), "Companion path escapes root")
    for part in (path, *path.parents):
        if part == root.parent:
            break
        require(not part.is_symlink() and not part.is_junction(), "Linked companion path")
    return path


@dataclass(frozen=True)
class SceneQualityPair:
    source_collection: str
    tile_id: str
    item_id: str
    spectral_10m: CatalogEntry
    spectral_10m_index: int
    spectral_20m: CatalogEntry
    spectral_20m_index: int
    quality_20m: CatalogEntry | None = None
    quality_20m_index: int | None = None

    @property
    def quality_available(self):
        return self.quality_20m is not None


class QualityCompanion:
    """Validate once at load; window requests reuse the cached identity indices."""

    def __init__(self, spectral_root, quality_root, spectral_entries):
        self.spectral_root = Path(spectral_root).resolve()
        self.quality_root = Path(quality_root).resolve()
        self.marker_path = self.quality_root / "companion.json"
        initial_stat = self._stat()
        before = self.marker_path.read_bytes()
        marker = json.loads(before)
        require(marker.get("state") == "published", "Quality companion is updating/unpublished")
        require(marker.get("contract_version") == 1 and marker.get("zarr_format") == 3 and
                marker.get("consolidated") is False, "Unsupported companion contract/metadata policy")
        for name in ("zarr", "xarray", "numcodecs"):
            require(marker["environment"][name] == importlib.metadata.version(name), f"Quality environment differs: {name}")
        require(digest(self.spectral_root / "catalog.parquet") == marker["binding"]["spectral_catalog_sha256"],
                "Spectral Catalog no longer matches companion")
        for relative, expected in marker["artifacts"].items():
            require(digest(child(self.quality_root, relative)) == expected, f"Published artifact differs: {relative}")
        require("catalog.parquet" in marker["artifacts"] and "quality_links.parquet" in marker["artifacts"],
                "Publication does not bind Catalog and links")
        quality_entries = load_catalog(self.quality_root, "sentinel-2")
        links = pq.read_table(self.quality_root / "quality_links.parquet").to_pylist()
        require(len(links) == len(quality_entries) == marker["accepted_scenes"], "Published scene counts differ")
        require(sum(marker["accepted_prefixes"].values()) == len(links), "Published prefix counts differ")
        # Each group has its own UTC->index map; a root sidecar is an identity
        # witness, not a reason to assume all native group indices are equal.
        group_cache = {}

        def location(entry):
            root_path = self.quality_root if entry.zarr_path.resolve().is_relative_to(self.quality_root) else self.spectral_root
            require(entry.zarr_path.resolve().is_relative_to(root_path), "Catalog store escapes root")
            cache_key = (entry.zarr_path, entry.zarr_group)
            if cache_key not in group_cache:
                root = zarr.open_group(str(entry.zarr_path), mode="r", use_consolidated=False, zarr_format=3)
                require(root.metadata.zarr_format == 3, "Expected format 3")
                group = root[entry.zarr_group]
                times = np.asarray(group["time"][:]).astype("datetime64[ns]")
                require(len(np.unique(times)) == len(times), "Duplicate UTC is unsupported")
                sidecar = json.loads((entry.zarr_path / "s2grits_resume_state.json").read_text(encoding="utf-8"))["items"]
                witnesses = {row["item_id"]: row for row in sidecar}
                require(len(witnesses) == len(sidecar), "Duplicate sidecar scene")
                require(np.array_equal(root["time"][:], np.asarray([stamp(row["datetime"]) for row in sidecar])),
                        "Root time/sidecar mismatch")
                group_cache[cache_key] = (group, {value: index for index, value in enumerate(times)}, witnesses)
            group, times, witnesses = group_cache[cache_key]
            witness = witnesses.get(entry.row["item_id"])
            require(witness is not None and stamp(witness["datetime"]) == stamp(entry.row["datetime"]), "Sidecar identity/UTC differs")
            require(stamp(entry.row["datetime"]) in times, "Catalog scene absent from group time axis")
            attrs = group.attrs
            require(attrs["grid_id"] == entry.grid_id and attrs["crs"] == entry.crs and
                    attrs["width"] == entry.width and attrs["height"] == entry.height and
                    np.allclose(attrs["transform"][:6], entry.transform, rtol=0, atol=1e-9), "Catalog/native grid differs")
            if entry.zarr_path.resolve().is_relative_to(self.quality_root):
                count = marker["accepted_prefixes"][entry.tile_id]
                require(group["scl"].shape == (count, entry.height, entry.width) and
                        group["scl"].dtype == np.dtype("uint8") and group["scl"].fill_value == 0 and
                        len(times) == len(witnesses) == count, "Quality length/dtype/fill differs from published prefix")
            return times[stamp(entry.row["datetime"])]

        spectral = {}
        self._at = {}
        for entry in spectral_entries:
            if entry.zarr_group not in {"10m", "20m"}:
                continue
            identity = key(entry)
            group_key = (*identity, entry.zarr_group)
            require(group_key not in spectral, "Spectral identity/group ambiguity")
            index = location(entry)
            spectral[group_key] = (entry, index)
            at = (entry.tile_id, entry.zarr_group, index)
            require(at not in self._at, "Multiple spectral stores/collections need explicit selection")
            self._at[at] = identity
        self._pairs = {}
        self._identities = {}
        for identity in {group_key[:3] for group_key in spectral}:
            require((*identity, "10m") in spectral and (*identity, "20m") in spectral, "Missing spectral native group")
            ten, ten_index = spectral[(*identity, "10m")]
            twenty, twenty_index = spectral[(*identity, "20m")]
            require(stamp(ten.row["datetime"]) == stamp(twenty.row["datetime"]), "Spectral group UTC differs")
            self._pairs[identity] = SceneQualityPair(*identity, ten, ten_index, twenty, twenty_index)
            self._identities.setdefault((identity[1], identity[2]), []).append(identity)
        quality = {}
        for entry in quality_entries:
            identity = key(entry)
            require(identity not in quality and entry.bands == ("scl",) and entry.zarr_group == "20m" and
                    entry.row["product_variant"] == "quality_scl_v1", "Invalid quality Catalog identity/asset")
            quality[identity] = (entry, location(entry))
        linked = set()
        for row in links:
            identity = (row["source_collection"], row["tile_id"], row["item_id"])
            require(identity not in linked and identity in quality and identity in self._pairs, "Orphan/duplicate quality link")
            linked.add(identity)
            pair = self._pairs[identity]
            q_entry, q_index = quality[identity]
            require(stamp(row["datetime"]) == stamp(q_entry.row["datetime"]) == stamp(pair.spectral_20m.row["datetime"]), "Link UTC differs")
            for prefix, entry, index, root in (
                ("spectral_10m", pair.spectral_10m, pair.spectral_10m_index, self.spectral_root),
                ("spectral_20m", pair.spectral_20m, pair.spectral_20m_index, self.spectral_root),
                ("quality_20m", q_entry, q_index, self.quality_root),
            ):
                require(child(root, row[prefix+"_store"]).resolve() == entry.zarr_path.resolve() and
                        row[prefix+"_group"] == entry.zarr_group and row[prefix+"_time_index"] == index and
                        row[prefix+"_grid_id"] == entry.grid_id, "Link group/index/grid differs")
            twenty = pair.spectral_20m
            require(q_entry.grid_id == twenty.grid_id and q_entry.crs == twenty.crs and
                    (q_entry.height, q_entry.width) == (twenty.height, twenty.width) and
                    np.allclose(q_entry.transform, twenty.transform, rtol=0, atol=1e-9), "Quality/spectral 20m grid differs")
            receipt_name = f"_task_state/accepted/{row['tile_id']}_{row['item_id']}.json"
            require(receipt_name in marker["artifacts"], "Accepted receipt is not published")
            receipt = json.loads(child(self.quality_root, receipt_name).read_text(encoding="utf-8"))
            verification = receipt["verification"]
            require(receipt["status"] == "ACCEPTED" and receipt["link"] == row and
                    receipt["source_sha256"] == row["source_sha256"] and verification["full_pixel_comparison"] and
                    verification["mismatched_pixels"] == 0 and verification["compared_pixels"] == q_entry.width*q_entry.height,
                    "Accepted full-pixel evidence differs")
            self._pairs[identity] = replace(pair, quality_20m=q_entry, quality_20m_index=q_index)
        require(linked == set(quality), "Quality link coverage differs")
        require(self.marker_path.read_bytes() == before and self._stat() == initial_stat,
                "Publication changed during load")
        self._marker_stat = initial_stat
        self.generation = marker["generation"]

    def _stat(self):
        value = self.marker_path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ino

    def ensure_current(self):
        # One cheap metadata operation; never rehash Catalogs per window.
        require(self._stat() == self._marker_stat, "Quality publication changed; construct a new Resolver")

    def scene(self, tile_id, item_id, *, collection=None, allow_missing=False):
        self.ensure_current()
        identities = ([(collection, tile_id, item_id)] if collection is not None
                      else self._identities.get((tile_id, item_id), []))
        require(len(identities) == 1 and identities[0] in self._pairs, "Scene identity absent/ambiguous")
        pair = self._pairs[identities[0]]
        require(allow_missing or pair.quality_available, "No published quality record for this scene")
        return pair

    def scene_at(self, tile_id, group, index):
        identity = self._at.get((tile_id, group, index))
        require(identity is not None, "Spectral group/index absent")
        return self.scene(tile_id, identity[2], collection=identity[0])
