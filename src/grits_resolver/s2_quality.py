"""Read-only, generation-bound S2 spectral/SCL scene pairing."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog, map_root_path


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


def group_scene_indices(entry):
    """Resolve native scene identities; UTC alone is not a unique observation."""
    root = zarr.open_group(str(entry.zarr_path), mode="r", use_consolidated=False)
    group = root[entry.zarr_group] if entry.zarr_group else root
    times = np.asarray(group["time"][:]).astype("datetime64[ns]")
    sidecar_path = entry.zarr_path / "s2grits_resume_state.json"
    if not sidecar_path.exists():
        return times, None
    rows = json.loads(sidecar_path.read_text(encoding="utf-8"))["items"]
    witnesses = {row["item_id"]: row for row in rows}
    require(len(witnesses) == len(rows), "Duplicate sidecar scene")
    root_times = np.asarray([stamp(row["datetime"]) for row in rows])
    require(np.array_equal(root["time"][:], root_times), "Root time/sidecar mismatch")
    if "item_id" in group:
        ids = [str(value) for value in group["item_id"][:]]
        require(len(ids) == len(times) and len(set(ids)) == len(ids), "Group identity axis differs")
    elif len(np.unique(times)) == len(times):
        by_time = {}
        for row in rows:
            when = stamp(row["datetime"])
            require(when not in by_time, "Duplicate sidecar UTC needs a group identity axis")
            by_time[when] = row["item_id"]
        require(all(when in by_time for when in times), "Group time absent from commit records")
        ids = [by_time[when] for when in times]
    else:
        # Native writer commits every listed group in root-sidecar order.
        require(np.array_equal(times, root_times) and all(
            entry.zarr_group in row.get("resolution_groups", {}) for row in rows),
            "Duplicate UTC needs native group commit identities")
        ids = [row["item_id"] for row in rows]
    result = {}
    for index, item in enumerate(ids):
        witness = witnesses.get(item)
        require(witness is not None and stamp(witness["datetime"]) == times[index],
                "Group identity/UTC differs from commit records")
        result[item] = (index, witness)
    return times, result


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


def file_state(path):
    try:
        value = Path(path).stat()
    except FileNotFoundError:
        return None
    return value.st_size, value.st_mtime_ns, value.st_ino


def public_receipt(receipt):
    from copy import deepcopy
    result = deepcopy(receipt)
    for binding in (result.get("binding", {}), result.get("write_intent", {}).get("binding", {})):
        for name in ("spectral_root", "quality_root", "cache_root"):
            binding.pop(name, None)
    return result



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
    unavailable_reason: str | None = None

    @property
    def quality_available(self):
        return self.quality_20m is not None


class QualityCompanion:
    """Validate once at load; window requests reuse the cached identity indices."""

    def __init__(self, spectral_root, quality_root, spectral_entries, *, pairing="legacy", root_map=None):
        require(pairing in {"legacy", "current"}, "pairing must be legacy or current")
        self.pairing = pairing
        self._views = []
        self._file_states = {}
        self._store_states = {}
        self.spectral_root = Path(spectral_root).resolve()
        self.quality_root = Path(quality_root).resolve()
        self.marker_path = self.quality_root / "companion.json"
        self._check_pending()
        for root in (self.spectral_root, self.quality_root):
            for name in ("catalog.parquet", "provenance/current.json"):
                path = root / name
                self._file_states[path] = file_state(path)
        initial_stat = self._stat()
        before = self.marker_path.read_bytes()
        marker = json.loads(before)
        require(marker.get("state") == "published", "Quality companion is updating/unpublished")
        require(marker.get("contract_version") == 1 and marker.get("zarr_format") == 3 and
                marker.get("consolidated") is False, "Unsupported companion contract/metadata policy")
        for name in ("zarr", "xarray", "numcodecs"):
            require(marker["environment"][name] == importlib.metadata.version(name), f"Quality environment differs: {name}")
        for name, root in (("spectral_root", self.spectral_root), ("quality_root", self.quality_root)):
            recorded = marker["binding"].get(name)
            require(recorded is None or map_root_path(recorded, root_map).resolve() == root,
                    "Moved companion root requires an explicit root_map")
        for relative, expected in marker["artifacts"].items():
            require(digest(child(self.quality_root, relative)) == expected, f"Published artifact differs: {relative}")
        require("catalog.parquet" in marker["artifacts"] and "quality_links.parquet" in marker["artifacts"],
                "Publication does not bind Catalog and links")
        self._file_states[self.quality_root / "quality_links.parquet"] = file_state(self.quality_root / "quality_links.parquet")
        self._pairing_document = None
        self._pairing_evidence = {}
        if pairing == "current":
            links = self._load_pairing(marker)
            binding = self._pairing_document["binding"]
        else:
            links = pq.read_table(self.quality_root / "quality_links.parquet").to_pylist()
            binding = marker["binding"]
        require(digest(self.spectral_root / "catalog.parquet") == binding["spectral_catalog_sha256"],
                "Spectral Catalog no longer matches companion/pairing")
        quality_entries = load_catalog(self.quality_root, "sentinel-2", root_map=root_map)
        old_links = pq.read_table(self.quality_root / "quality_links.parquet").to_pylist()
        require(len(old_links) == len(quality_entries) == marker["accepted_scenes"], "Published scene counts differ")
        require(sum(marker["accepted_prefixes"].values()) == len(old_links), "Published prefix counts differ")
        old_by_identity = {(r["source_collection"], r["tile_id"], r["item_id"]): r for r in old_links}
        require(len(old_by_identity) == len(old_links), "Duplicate legacy quality identity")
        # Legacy unique-UTC groups may reorder scenes; native duplicate-UTC
        # groups require the explicit commit-order identity contract.
        group_cache = {}

        def location(entry):
            root_path = self.quality_root if entry.zarr_path.resolve().is_relative_to(self.quality_root) else self.spectral_root
            require(entry.zarr_path.resolve().is_relative_to(root_path), "Catalog store escapes root")
            cache_key = (entry.zarr_path, entry.zarr_group)
            if cache_key not in group_cache:
                root = zarr.open_group(str(entry.zarr_path), mode="r", use_consolidated=False, zarr_format=3)
                require(root.metadata.zarr_format == 3, "Expected format 3")
                group = root[entry.zarr_group]
                times, identities = group_scene_indices(entry)
                require(identities is not None, "Quality pairing requires scene commit records")
                group_cache[cache_key] = (group, times, identities)
                self._watch_store(entry.zarr_path, root)
            group, times, identities = group_cache[cache_key]
            found = identities.get(entry.row["item_id"])
            require(found is not None, "Catalog scene absent from group identities")
            index, witness = found
            require(stamp(witness["datetime"]) == stamp(entry.row["datetime"]) and
                    witness.get("collection", entry.row["source_collection"]) == entry.row["source_collection"],
                    "Sidecar identity/UTC differs")
            attrs = group.attrs
            require(attrs["grid_id"] == entry.grid_id and attrs["crs"] == entry.crs and
                    attrs["width"] == entry.width and attrs["height"] == entry.height and
                    np.allclose(attrs["transform"][:6], entry.transform, rtol=0, atol=1e-9), "Catalog/native grid differs")
            if entry.zarr_path.resolve().is_relative_to(self.quality_root):
                count = marker["accepted_prefixes"][entry.tile_id]
                require(group["scl"].shape == (count, entry.height, entry.width) and
                        group["scl"].dtype == np.dtype("uint8") and group["scl"].fill_value == 0 and
                        len(times) == len(identities) == count, "Quality length/dtype/fill differs from published prefix")
            return index

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
            self._at.setdefault(at, []).append(identity)
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
        receipts = {}
        for identity, (entry, _) in quality.items():
            row = old_by_identity.get(identity)
            require(row is not None, "Quality Catalog absent from legacy links")
            receipt_name = f"_task_state/accepted/{identity[1]}_{identity[2]}.json"
            require(receipt_name in marker["artifacts"], "Accepted receipt is not published")
            receipt = json.loads(child(self.quality_root, receipt_name).read_text(encoding="utf-8"))
            verification = receipt["verification"]
            require(receipt["status"] == "ACCEPTED" and receipt["link"] == row and
                    receipt["source_sha256"] == row["source_sha256"] and verification["full_pixel_comparison"] and
                    verification["mismatched_pixels"] == 0 and verification["compared_pixels"] == entry.width*entry.height,
                    "Accepted full-pixel evidence differs")
            receipts[identity] = receipt
        for row in links:
            identity = (row["source_collection"], row["tile_id"], row["item_id"])
            require(identity not in linked and identity in self._pairs, "Orphan/duplicate quality link")
            linked.add(identity)
            pair = self._pairs[identity]
            require(stamp(row["datetime"]) == stamp(pair.spectral_20m.row["datetime"]), "Link UTC differs")
            self._check_link(row, "spectral_10m", pair.spectral_10m, pair.spectral_10m_index, self.spectral_root)
            self._check_link(row, "spectral_20m", pair.spectral_20m, pair.spectral_20m_index, self.spectral_root)
            available = row.get("quality_available", True) if pairing == "current" else True
            require(type(available) is bool, "quality_available must be boolean")
            if not available:
                require(isinstance(row.get("unavailable_reason"), str) and row["unavailable_reason"],
                        "Unavailable quality needs a reason")
                self._pairs[identity] = replace(pair, unavailable_reason=row["unavailable_reason"])
            if identity in quality:
                q_entry, q_index = quality[identity]
                self._check_link(row, "quality_20m", q_entry, q_index, self.quality_root)
                receipt = receipts[identity]
                if pairing == "current":
                    proof = self._pairing_evidence.get(identity[1] + "_" + identity[2])
                    require(proof is not None and proof.get("legacy_receipt_sha256") ==
                            marker["artifacts"][f"_task_state/accepted/{identity[1]}_{identity[2]}.json"]
                            and proof["receipt"] == public_receipt(receipt), "Pairing receipt evidence differs")
                    require(row.get("quality_generation_id") == receipt.get("generation_id"), "Quality generation differs")
                if available:
                    require(receipt["source_sha256"] == row["source_sha256"], "Quality source version differs")
                    require(stamp(row["datetime"]) == stamp(q_entry.row["datetime"]), "Quality UTC differs")
                    twenty = pair.spectral_20m
                    require(q_entry.grid_id == twenty.grid_id and q_entry.crs == twenty.crs and
                            (q_entry.height, q_entry.width) == (twenty.height, twenty.width) and
                            np.allclose(q_entry.transform, twenty.transform, rtol=0, atol=1e-9), "Quality/spectral 20m grid differs")
                    self._pairs[identity] = replace(pair, quality_20m=q_entry, quality_20m_index=q_index)
            else:
                require(not available, "Available quality has no accepted scene")
        if pairing == "legacy":
            require(linked == set(quality), "Quality link coverage differs")
            from .provenance import ProvenanceView
            self._views = [(ProvenanceView(root), root) for root in (self.spectral_root, self.quality_root)]
        else:
            self._verify_pairing_revisions(links, old_links, receipts)
        require(self.marker_path.read_bytes() == before and self._stat() == initial_stat,
                "Publication changed during load")
        self._marker_stat = initial_stat
        self.generation = marker["generation"]
        self.ensure_current()

    def _load_pairing(self, marker):
        from .provenance import canonical_json
        pointer_path = self.quality_root / "provenance/pairings/current.json"
        self._file_states[pointer_path] = file_state(pointer_path)
        require(pointer_path.is_file(), "No independent quality pairing is published")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        require(pointer.get("kind") == "independent_quality_pairing" and pointer.get("contract_version") == 1,
                "Unsupported independent pairing contract")
        require(isinstance(pointer.get("run_id"), str) and
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", pointer["run_id"]), "Unsafe pairing run identity")
        directory = child(self.quality_root, "provenance/pairings/" + pointer["run_id"])
        require(set(pointer["artifacts"]) == {"pairing.json", "quality_links.parquet", "evidence.json"},
                "Pairing must bind its document, links and evidence")
        for name, expected in pointer["artifacts"].items():
            path = child(directory, name)
            self._file_states[path] = file_state(path)
            require(digest(path) == expected, "Published pairing artifact differs")
        document = json.loads((directory / "pairing.json").read_text(encoding="utf-8"))
        evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
        links = pq.read_table(directory / "quality_links.parquet").to_pylist()
        require(document.get("contract_version") == 1 and all(document.get(k) == pointer[k] for k in
                ("run_id", "spectral_data_revision", "quality_data_revision")), "Pairing pointer/document differs")
        require(hashlib.sha256(canonical_json({**document, "links": links, "evidence": evidence})).hexdigest() ==
                pointer["request_sha256"], "Pairing payload digest differs")
        require(digest(self.marker_path) == document["legacy_companion_sha256"], "Legacy quality version changed")
        self._pairing_document, self._pairing_evidence = document, evidence
        return links

    def _verify_pairing_revisions(self, links, old_links, receipts):
        from .provenance import (
            ProvenanceView,
            canonical_json,
            data_revision,
            read_snapshot,
        )
        document = self._pairing_document
        spectral = ProvenanceView(self.spectral_root)
        self._views.append((spectral, self.spectral_root))
        if document["spectral_revision_scope"].startswith("published_"):
            require(spectral.available and spectral.data_revision == document["spectral_data_revision"],
                    "Spectral data revision changed")
        else:
            require(document["spectral_revision_scope"] == "historical_structure_only" and not spectral.available,
                    "Unsupported or changed spectral revision scope")
            witnesses = {"catalog.parquet": digest(self.spectral_root / "catalog.parquet")}
            locations = []
            for link in links:
                for group in ("10m", "20m"):
                    prefix = "spectral_" + group + "_"
                    location = {name: link[prefix + name] for name in ("store", "group", "time_index", "grid_id")}
                    sidecar = location["store"] + "/s2grits_resume_state.json"
                    witnesses[sidecar] = digest(child(self.spectral_root, sidecar))
                    locations.append({"item_id": link["item_id"], **location})
            revision = "observed-" + hashlib.sha256(canonical_json({"witnesses": witnesses, "locations": locations})).hexdigest()
            require(revision == document["spectral_data_revision"], "Historical spectral revision changed")
        quality_snapshot = read_snapshot(self.quality_root)
        if document["quality_revision_scope"] == "production_snapshot":
            quality = ProvenanceView(self.quality_root, required=True)
            self._views.append((quality, self.quality_root))
            require(quality.data_revision == document["quality_data_revision"], "Quality data revision changed")
        else:
            require(document["quality_revision_scope"] == "historical_or_changed_observation",
                    "Unsupported quality revision scope")
        outputs, witnesses = [], {}
        for link in old_links:
            identity = (link["source_collection"], link["tile_id"], link["item_id"])
            receipt = receipts[identity]
            logical = {"collection": identity[0], "tile_id": identity[1], "item_id": identity[2], "group": "20m",
                       "grid_id": link["quality_20m_grid_id"], "variables": ["scl"]}
            outputs.append({"output_id": "output-" + hashlib.sha256(canonical_json(logical)).hexdigest(),
                "generation_id": receipt.get("generation_id"), "store_path": link["quality_20m_store"],
                "zarr_group": "20m", "variable_scope": ["scl"], "tile_id": identity[1], "geometry_group_id": None,
                "grid_id": link["quality_20m_grid_id"], "product_type": "quality_scl", "record_kind": "scene",
                "time_index": link["quality_20m_time_index"], "time_value_utc": link["datetime"],
                "period_start": None, "period_end": None, "source_item_id": identity[2]})
            sidecar = link["quality_20m_store"] + "/s2grits_resume_state.json"
            witnesses[sidecar] = digest(child(self.quality_root, sidecar))
            decoded = receipt["verification"].get("decoded_sha256")
            if decoded:
                witnesses[f"decoded/{link['quality_20m_store']}/20m/scl/{identity[2]}"] = decoded
        for link in links:
            identity = (link["source_collection"], link["tile_id"], link["item_id"])
            proof = self._pairing_evidence.get(identity[1] + "_" + identity[2])
            if not proof:
                continue
            integrity = proof["current_integrity"]
            expected = integrity.get("decoded_sha256")
            if expected:
                require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid pixel evidence digest")
                witness_key = f"decoded/{link['quality_20m_store']}/20m/scl/{identity[2]}"
                observed = (quality.snapshot["witnesses"].get(witness_key)
                            if document["quality_revision_scope"] == "production_snapshot"
                            else expected)
                # Pixel verification belongs to the signed-off producer; reads check its evidence and current structure.
                require(observed == expected, "Quality pixel evidence differs from the published revision")
                witnesses[witness_key] = expected
            if link["quality_available"]:
                require(expected and integrity.get("integrity") in
                        {"accepted_decoded_digest", "original_COG_full_pixel_comparison"},
                        "Available quality lacks checked pixel evidence")
        require(data_revision(outputs, witnesses) == document["quality_data_revision"], "Quality observation revision differs")
        if document["quality_revision_scope"] == "production_snapshot":
            require(quality_snapshot["pointer"]["data_revision"] == document["quality_data_revision"],
                    "Quality publication revision differs")

    def _stat(self):
        value = self.marker_path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ino

    def _check_pending(self):
        for root in (self.spectral_root, self.quality_root):
            require(not list((root / "provenance/pending").glob("*.json")), "Dataset has an unfinished provenance write")

    def _watch_store(self, path, root):
        if path.resolve() in self._store_states:
            return
        paths = {path / "zarr.json", path / "s2grits_resume_state.json"}
        for name, group in [("", root), *list(root.groups())]:
            base = path / name if name else path
            paths.add(base / "zarr.json")
            for variable in group.array_keys():
                paths.add(base / variable / "zarr.json")
                if variable in {"time", "x", "y", "item_id"}:
                    paths.update(p for p in (base / variable).rglob("*") if p.is_file())
        self._store_states[path.resolve()] = {p: file_state(p) for p in paths}

    @staticmethod
    def _check_link(row, prefix, entry, index, root):
        require(child(root, row[prefix + "_store"]).resolve() == entry.zarr_path.resolve() and
                row[prefix + "_group"] == entry.zarr_group and row[prefix + "_time_index"] == index and
                row[prefix + "_grid_id"] == entry.grid_id, "Link group/index/grid differs")

    def ensure_current(self, pair=None):
        self._check_pending()
        require(self._stat() == self._marker_stat, "Quality publication changed; construct a new Resolver")
        for path, expected in self._file_states.items():
            require(file_state(path) == expected, "Quality pairing/data version changed; construct a new Resolver")
        entries = [] if pair is None else [pair.spectral_10m, pair.spectral_20m, pair.quality_20m]
        stores = {entry.zarr_path.resolve() for entry in entries if entry is not None}
        for store in stores:
            for path, expected in self._store_states[store].items():
                require(file_state(path) == expected, "Selected store changed; construct a new Resolver")
        for view, root in self._views:
            view.assert_current(stores=[p for p in stores if p.is_relative_to(root)])

    def scene(self, tile_id, item_id, *, collection=None, allow_missing=False):
        self.ensure_current()
        identities = ([(collection, tile_id, item_id)] if collection is not None
                      else self._identities.get((tile_id, item_id), []))
        require(len(identities) == 1 and identities[0] in self._pairs, "Scene identity absent/ambiguous")
        pair = self._pairs[identities[0]]
        self.ensure_current(pair)
        require(allow_missing or pair.quality_available, "No published quality record for this scene")
        return pair

    def scene_at(self, tile_id, group, index):
        identities = self._at.get((tile_id, group, index), [])
        require(len(identities) == 1, "Spectral group/index absent or ambiguous; select item_id")
        identity = identities[0]
        return self.scene(tile_id, identity[2], collection=identity[0])
