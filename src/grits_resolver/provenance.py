"""Read-only v1/v2 releases over v1 tables, with explicit reload guards."""
from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import zarr

from .catalog import CatalogConsistencyError
from .provenance_contract import (
    canonical_json,
    data_revision,
    read_tables,
    validate_numeric_processing,
    validate_historical_annotations,
)


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _io_path(path):
    value = os.path.abspath(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def safe_relative(root, relative):
    if (not isinstance(relative, str) or not relative or relative == "."
            or any(c in relative for c in '\\:?*<>|"')
            or any(ord(c) < 32 or ord(c) == 127 for c in relative)):
        raise CatalogConsistencyError("Invalid relative provenance path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise CatalogConsistencyError("Provenance path escapes root")
    root = _io_path(root)
    target = root.joinpath(*pure.parts)
    current = target
    while current != root:
        if current.is_symlink() or current.is_junction():
            raise CatalogConsistencyError("Linked provenance/data paths are not portable")
        current = current.parent
    if not target.resolve().is_relative_to(root.resolve()):
        raise CatalogConsistencyError("Provenance path escapes root")
    return target


def read_json(path):
    return json.loads(_io_path(path).read_text(encoding="utf-8"))


def _pointer(root):
    path = safe_relative(root, "provenance/current.json")
    if not path.exists():
        return None
    value = read_json(path)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] not in (1, 2)
            or not re.fullmatch(r"meta-[0-9a-f]{64}", str(value.get("metadata_release_id", "")))
            or not re.fullmatch(r"data-[0-9a-f]{64}", str(value.get("data_revision", "")))
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, "")))
                   for key in ("dataset_sha256", "manifest_sha256"))):
        raise CatalogConsistencyError("Unsupported or invalid provenance pointer; supported release versions: 1, 2. Upgrade GRiTS-Resolver.")
    return value


def _walk_error(error):
    raise error


def _files(root):
    result = {}
    for folder, dirs, files in os.walk(root, onerror=_walk_error):
        for name in dirs + files:
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            safe_relative(root, relative)
            if name in files:
                result[relative] = path
    return result


def read_snapshot(root):
    """Verify metadata bytes, schema, references and revision; no imagery checks."""
    try:
        pointer = _pointer(root)
        if pointer is None:
            return {"pointer": None, "dataset": None, "outputs": [], "sources": [],
                    "source_links": [], "variables": [], "limitations": [], "witnesses": {}, "evidence": {}}
        directory = safe_relative(root, "provenance/releases/" + pointer["metadata_release_id"])
        manifest = directory / "FILES.sha256"
        if file_digest(manifest) != pointer["manifest_sha256"]:
            raise CatalogConsistencyError("Provenance manifest hash mismatch")
        listed = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest) or name in listed or name == "FILES.sha256":
                raise CatalogConsistencyError("Invalid provenance manifest")
            safe_relative(directory, name)
            listed[name] = digest
        actual = {name: file_digest(path) for name, path in _files(directory).items() if name != "FILES.sha256"}
        if actual != listed or actual.get("dataset.json") != pointer["dataset_sha256"]:
            raise CatalogConsistencyError("Provenance release differs from its manifest")
        dataset = read_json(directory / "dataset.json")
        if (type(dataset.get("schema_version")) is not int
                or any(dataset.get(key) != pointer[key] for key in ("schema_version", "metadata_release_id", "data_revision"))
                or dataset.get("delivery_kind") != "metadata_only"):
            raise CatalogConsistencyError("Provenance dataset and pointer disagree")
        outputs, sources, links = read_tables(directory)
        for row in outputs + sources + links:
            if row.get("evidence_ref") and row["evidence_ref"] not in listed:
                raise CatalogConsistencyError("Missing provenance evidence reference")
        witnesses = read_json(directory / "evidence/data_witnesses.json")
        if data_revision(outputs, witnesses) != pointer["data_revision"]:
            raise CatalogConsistencyError("Provenance data revision differs from output mapping")
        if dataset.get("counts") != {"outputs": len(outputs), "sources": len(sources), "source_links": len(links)}:
            raise CatalogConsistencyError("Provenance table counts disagree")
        result = {"pointer": pointer, "dataset": dataset, "outputs": outputs, "sources": sources,
                  "source_links": links, "witnesses": witnesses,
                  "variables": read_json(directory / "variables.json"),
                  "limitations": read_json(directory / "limitations.json"),
                  "evidence": {name[len("evidence/"):]: read_json(directory / name) for name in listed
                               if name.startswith("evidence/") and name != "evidence/data_witnesses.json"}}
        if pointer["schema_version"] == 2:
            validate_historical_annotations(outputs, result["variables"], result["limitations"],
                                            result["evidence"], required=True)
        if _pointer(root) != pointer:
            raise CatalogConsistencyError("Provenance version changed during load; reload explicitly")
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CatalogConsistencyError("Cannot validate provenance snapshot: " + str(exc)) from exc


def _stamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise CatalogConsistencyError("Missing time cannot locate an output")
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def _times(array):
    values = np.asarray(array[:])
    if values.ndim != 1:
        raise CatalogConsistencyError("Time coordinate must be one-dimensional")
    if values.dtype.kind in "iu":
        units = str(array.attrs.get("units", ""))
        match = re.fullmatch(r"(nanoseconds|microseconds|milliseconds|seconds|minutes|hours|days) since (.+)", units)
        if match:
            unit = {"nanoseconds": "ns", "microseconds": "us", "milliseconds": "ms", "seconds": "s", "minutes": "m", "hours": "h", "days": "D"}[match[1]]
            values = pd.Timestamp(match[2]) + pd.to_timedelta(values, unit=unit)
        elif units:
            raise CatalogConsistencyError("Unsupported time units")
        else:
            values = values.astype("datetime64[ns]")
    return [_stamp(value) for value in values]


def _stat(path):
    try:
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    except FileNotFoundError:
        return None


def _pending(root):
    path = safe_relative(root, "provenance/pending")
    try:
        with os.scandir(path) as entries:
            if any(entry.name.endswith(".json") for entry in entries):
                raise CatalogConsistencyError("Unfinished provenance write is pending; finish production before reading")
    except FileNotFoundError:
        return


class ProvenanceView:
    """A frozen loaded version. Construct a new view after a producer completes.

    required checks availability, not the completeness of historical evidence.
    Lightweight guards assume cooperating writers; they are not pixel hashes.
    """
    def __init__(self, data_root, *, metadata_root=None, required=False):
        self.data_root = _io_path(data_root)
        self.metadata_root = _io_path(metadata_root or data_root)
        self._global = {}
        self._store_files = {}
        self._groups = {}
        try:
            for root in {self.data_root, self.metadata_root}:
                _pending(root)
            self._initial_current = {root: _pointer(root) for root in {self.data_root, self.metadata_root}}
            for path in [self.data_root / "catalog.parquet", *self.data_root.glob("*/catalog.parquet")]:
                self._global[path] = _stat(path)
            self.snapshot = read_snapshot(self.metadata_root)
            self.pointer = self.snapshot["pointer"]
            self.available = self.pointer is not None
            self.data_revision = self.pointer["data_revision"] if self.pointer else None
            if required and not self.available:
                raise CatalogConsistencyError("Required provenance metadata is missing")
            if self.available:
                release = safe_relative(self.metadata_root, "provenance/releases/" + self.pointer["metadata_release_id"])
                for name in ("FILES.sha256", "dataset.json", "outputs.parquet", "sources.parquet", "source_links.parquet", "variables.json", "limitations.json"):
                    self._watch(release / name)
                data_pointer = self._initial_current[self.data_root]
                if data_pointer and data_pointer["data_revision"] != self.data_revision:
                    raise CatalogConsistencyError("Detached metadata describes a different data revision")
                self._verify_sources()
            self._sources = {r["source_id"]: r for r in self.snapshot["sources"]}
            self._links = {}
            for link in self.snapshot["source_links"]:
                self._links.setdefault(link["output_id"], []).append(link)
            self.assert_current(stores=self._store_files)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise CatalogConsistencyError("Cannot bind provenance to data: " + str(exc)) from exc

    def _group(self, relative):
        if relative not in self._groups:
            self._groups[relative] = zarr.open_group(str(safe_relative(self.data_root, relative)), mode="r", use_consolidated=False)
        return self._groups[relative]

    def _store_for(self, relative):
        matches = [store for store in self._store_files if relative == store or relative.startswith(store + "/")]
        if len(matches) != 1:
            raise CatalogConsistencyError("Witness does not identify exactly one selected store")
        return matches[0]

    def _watch(self, path, store=None):
        target = self._global if store is None else self._store_files[store]
        target.setdefault(path, _stat(path))

    def _logical_group_digest(self, relative):
        group_path = relative.removesuffix("/zarr.json")
        group = self._group(group_path)
        times = np.asarray(group["time"][:]).astype("datetime64[ns]")
        layout = {name: {"shape": list(array.shape), "dtype": str(array.dtype), "chunks": list(array.chunks)}
                  for name, array in group.arrays()}
        return hash_bytes(canonical_json({"attrs": dict(group.attrs), "arrays": layout, "times": [str(t) for t in times]}))

    def _decoded_receipt_digest(self, name):
        location, item_id = name.removeprefix("decoded/").rsplit("/", 1)
        matches = [row for row in self.snapshot["outputs"]
                   if row["store_path"] + "/" + row["zarr_group"] + "/scl" == location
                   and row["source_item_id"] == item_id]
        if len(matches) != 1:
            raise CatalogConsistencyError("Decoded quality witness has no unique output")
        row = matches[0]
        ref = (row.get("evidence_ref") or "").removeprefix("evidence/")
        receipt = self.snapshot["evidence"].get(ref, {})
        verification = receipt.get("verification", {})
        link = receipt.get("link", {})
        if (receipt.get("status") != "ACCEPTED" or receipt.get("generation_id") != row["generation_id"]
                or not verification.get("full_pixel_comparison") or verification.get("mismatched_pixels") != 0
                or verification.get("decoded_hash_scheme") != "uint8-block512-v1"
                or link.get("quality_20m_store") != row["store_path"]
                or link.get("quality_20m_time_index") != row["time_index"]
                or link.get("quality_20m_grid_id") != row["grid_id"]
                or receipt.get("item_id") != item_id):
            raise CatalogConsistencyError("Quality decoded witness has no matching accepted receipt")
        return verification.get("decoded_sha256")

    def _verify_sources(self):
        for row in self.snapshot["outputs"]:
            self._store_files.setdefault(row["store_path"], {})
        dataset = self.snapshot["dataset"]
        logical_native = dataset.get("production_scope") == "s2_native_scenes"
        # Older P3 publications identify native S2 by production_run_id and scene rows.
        logical_native = logical_native or (bool(dataset.get("production_run_id")) and
            all(row["product_type"] == "scenes" for row in self.snapshot["outputs"]) and
            not any(key.endswith("#values") for key in self.snapshot["witnesses"]))
        for name, expected in self.snapshot["witnesses"].items():
            if name.startswith("decoded/"):
                actual = self._decoded_receipt_digest(name)
            elif name.endswith("#values"):
                relative = name.removesuffix("#values")
                group_path, variable = relative.rsplit("/", 1)
                array = self._group(group_path)[variable]
                values = _times(array) if variable == "time" else [str(v) for v in array[:]] if variable == "item_id" else np.asarray(array[:]).tolist()
                actual = hash_bytes(canonical_json(values))
                for path in _files(safe_relative(self.data_root, relative)).values():
                    self._watch(path, self._store_for(relative))
            else:
                path = safe_relative(self.data_root, name)
                store = next((s for s in self._store_files if name.startswith(s + "/")), None)
                if store is None and name.startswith("provenance/runs/"):
                    run_id = name.split("/")[2]
                    candidates = {r["store_path"] for r in self.snapshot["outputs"] if r["run_id"] == run_id}
                    store = next(iter(candidates)) if len(candidates) == 1 else None
                self._watch(path, store)
                actual = self._logical_group_digest(name) if logical_native and name.endswith("/zarr.json") else file_digest(path)
            if actual != expected:
                raise CatalogConsistencyError("Stale provenance data witness: " + name)
        times_by_group, arrays_by_group, commits = {}, {}, {}
        for row in self.snapshot["outputs"]:
            group_name = (row["zarr_group"] or "").strip("/")
            relative = row["store_path"] + ("/" + group_name if group_name else "")
            group = self._group(relative)
            if relative not in arrays_by_group:
                arrays_by_group[relative] = dict(group.arrays())
                times_by_group[relative] = _times(group["time"]) if "time" in group else []
                for parent in [relative, *[relative + "/" + v for v in arrays_by_group[relative]]]:
                    for filename in ("zarr.json", ".zattrs", ".zarray", ".zgroup"):
                        self._watch(safe_relative(self.data_root, parent + "/" + filename), row["store_path"])
                for coordinate in ("time", "x", "y", "item_id"):
                    if coordinate in arrays_by_group[relative]:
                        for path in _files(safe_relative(self.data_root, relative + "/" + coordinate)).values():
                            self._watch(path, row["store_path"])
            arrays, times = arrays_by_group[relative], times_by_group[relative]
            if row.get("grid_id") and group.attrs.get("grid_id", self._group(row["store_path"]).attrs.get("grid_id")) != row["grid_id"]:
                raise CatalogConsistencyError("Output grid differs from provenance")
            index = row["time_index"]
            if index is not None and (index >= len(times) or _stamp(times[index]) != _stamp(row["time_value_utc"])):
                raise CatalogConsistencyError("Output time/index differs from provenance")
            for variable in row["variable_scope"]:
                if variable not in arrays or arrays[variable].ndim != (2 if index is None else 3):
                    raise CatalogConsistencyError("Output variable layout differs from provenance")
                if index is not None and arrays[variable].shape[0] != len(times):
                    raise CatalogConsistencyError("Output variable has uncommitted time layers")
            if row.get("source_item_id"):
                if "item_id" in arrays:
                    if str(arrays["item_id"][index]) != row["source_item_id"]:
                        raise CatalogConsistencyError("Output scene identity differs from its group")
                else:
                    if row["store_path"] not in commits:
                        sidecar = next((safe_relative(self.data_root, row["store_path"] + "/" + name)
                                        for name in ("s2grits_resume_state.json", "s2grits_metadata.json")
                                        if safe_relative(self.data_root, row["store_path"] + "/" + name).exists()), None)
                        commits[row["store_path"]] = read_json(sidecar)["items"] if sidecar else []
                    items = commits[row["store_path"]]
                    if items:
                        possible = ([items[index]] if logical_native else
                                    [item for item in items if _stamp(item["datetime"]) == _stamp(times[index])])
                        if len(possible) != 1 or possible[0]["item_id"] != row["source_item_id"]:
                            raise CatalogConsistencyError("Output scene identity differs from commit record")
        op = dataset.get("production_run_id")
        plan = self.snapshot["evidence"].get("runs/" + str(op) + "/catalog_plan.json") if op else None
        if plan:
            bindings = ({entry["path"]: entry["after"] for entry in plan["entries"]} if "entries" in plan
                        else {name: entry["desired_sha256"] for name, entry in plan["targets"].items()})
            for name, expected in bindings.items():
                path = safe_relative(self.data_root, name)
                if file_digest(path) != expected:
                    raise CatalogConsistencyError("Current Catalog differs from published production")
                self._watch(path)
        self._groups.clear()

    def _relative_store(self, store):
        path = Path(store)
        if path.is_absolute():
            try:
                return _io_path(path).relative_to(self.data_root).as_posix()
            except ValueError as exc:
                raise CatalogConsistencyError("Requested store is outside bound dataset") from exc
        relative = str(store).replace("\\", "/")
        safe_relative(self.data_root, relative)
        return relative

    def assert_current(self, *, stores=()):
        """Cheap checks before AND after a materialized read; never rehash a store."""
        try:
            for root, pointer in self._initial_current.items():
                _pending(root)
                if _pointer(root) != pointer:
                    raise CatalogConsistencyError("Provenance version changed; reload explicitly")
            for path, expected in self._global.items():
                if _stat(path) != expected:
                    raise CatalogConsistencyError("Catalog or provenance evidence changed; reload explicitly")
            for store in stores:
                relative = self._relative_store(store)
                for path, expected in self._store_files.get(relative, {}).items():
                    if _stat(path) != expected:
                        raise CatalogConsistencyError("Selected store structure changed; reload explicitly")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CatalogConsistencyError("Cannot verify current provenance: " + str(exc)) from exc


    def _numeric_processing(self, row):
        ref = (row.get("evidence_ref") or "").removeprefix("evidence/")
        receipt = self.snapshot["evidence"].get(ref)
        result = {"output_id": row["output_id"], "generation_id": row["generation_id"],
                  "record": None, "unknown_reason": "No numerical processing record is attached to this output generation."}
        if not isinstance(receipt, dict) or "numeric_processing" not in receipt:
            return result
        if (not row["generation_id"] or receipt.get("output_id") != row["output_id"]
                or receipt.get("generation_id") != row["generation_id"]):
            raise CatalogConsistencyError("Numerical processing receipt differs from output/generation")
        record = receipt["numeric_processing"]
        try:
            validate_numeric_processing(record)
        except (ValueError, TypeError, KeyError) as exc:
            raise CatalogConsistencyError("Invalid numerical processing record: " + str(exc)) from exc
        return {**result, "record": record, "unknown_reason": None}

    def explain(self, *, output_id=None, store_path=None, zarr_group=None, time_index=None, variable=None):
        relative = self._relative_store(store_path) if store_path is not None else None
        self.assert_current(stores=[relative] if relative is not None else self._store_files)
        rows = [row for row in self.snapshot["outputs"]
                if (output_id is None or row["output_id"] == output_id)
                and (relative is None or row["store_path"] == relative)
                and (zarr_group is None or (row["zarr_group"] or "").strip("/") == zarr_group.strip("/"))
                and (time_index is None or row["time_index"] == time_index)
                and (variable is None or variable in row["variable_scope"])]
        matches = [{"output": row, "sources": [{"link": link, "source": self._sources[link["source_id"]]}
                                               for link in self._links.get(row["output_id"], [])]}
                   for row in rows]
        references = {r.get("evidence_ref", "").removeprefix("evidence/") for r in rows if r.get("evidence_ref")}
        references.update(r.get("evidence_ref", "").removeprefix("evidence/")
                          for match in matches for source in match["sources"]
                          for r in (source["link"], source["source"]) if r.get("evidence_ref"))
        runs = tuple("runs/" + row["run_id"] + "/" for row in rows if row.get("run_id"))
        evidence = {key: value for key, value in self.snapshot["evidence"].items()
                    if key in references or key.startswith(runs)}
        result = {"metadata_status": "available" if self.available else "missing", "matched": bool(matches),
                "dataset": self.snapshot["dataset"], "pointer": self.pointer, "outputs": matches,
                "variables": self.snapshot["variables"], "limitations": self.snapshot["limitations"],
                "evidence": evidence, "numeric_processing": [self._numeric_processing(row) for row in rows],
                "verification_scope": "release hashes, recorded acceptance and current structural witnesses; no fresh image chunk checksum",
                "unknown_reason": None if matches else "No provenance output matches this selection; no processing history is inferred."}
        result = deepcopy(result)

        selected = {row["output_id"]: row.get("generation_id") for row in rows}

        def selected_scope(scope):
            if (not isinstance(scope, dict) or any(
                    not isinstance(identifier, str) or (generation is not None and not isinstance(generation, str))
                    for identifier, generation in scope.items())):
                raise CatalogConsistencyError("Invalid historical annotation output generations")
            return {identifier: generation for identifier, generation in scope.items()
                    if identifier in selected and selected[identifier] == generation}

        for table, fields in (
            ("variables", ("units", "historical_processing_ref", "historical_processing_basis")),
            ("limitations", ("historical_processing_ref",)),
        ):
            for record in result[table]:
                if "historical_output_generations" not in record:
                    continue
                scope = selected_scope(record["historical_output_generations"])
                if scope:
                    record["historical_output_generations"] = scope
                else:
                    for field in (*fields, "historical_output_generations"):
                        record.pop(field, None)
        for value in result["evidence"].values():
            context = value.get("historical_context") if isinstance(value, dict) else None
            if isinstance(context, dict) and "output_generations" in context:
                scope = selected_scope(context["output_generations"])
                if scope:
                    context["output_generations"] = scope
                else:
                    value.pop("historical_context")
        self.assert_current(stores=[relative] if relative is not None else self._store_files)
        return result
