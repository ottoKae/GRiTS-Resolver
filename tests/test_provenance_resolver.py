"""Public reader guards and relocation; all rasters are tiny local fixtures."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr
from test_provenance import publish
from test_s1_product_tracks import _row, _write_dynamic, _write_static
from test_s2_quality_pairing import pair_case as _quality_fixture

from grits_resolver import CatalogConsistencyError, MultiSensorResolver, S1GRiTSResolver
from grits_resolver import resolver as resolver_module
from grits_resolver import s1 as s1_module
from grits_resolver.spatial import TargetWindow


def _snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def s1_case(tmp_path):
    root, tile = tmp_path / "s1", "17AAA"
    monthly, static = f"{tile}/monthly.zarr", f"{tile}/static.zarr"
    _write_dynamic(root / monthly)
    _write_static(root / static)
    rows = [_row(tile, "smonthly", monthly, 18, item="monthly-18"),
            _row(tile, "static", static, 18, item="static-18", reference=monthly)]
    pq.write_table(pa.Table.from_pylist(rows), root / "catalog.parquet")
    return {"root": root, "tile": tile, "monthly": monthly, "static": static, "rows": rows}


def _pair(resolver, case):
    return resolver.tracks(case["tile"], require_static=True, strict_static_reference=True)[0]


def _pending(root):
    path = root / "provenance/pending/unfinished.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "unfinished", "phase": "writing"}), encoding="utf-8")
    return path


def test_legacy_values_and_static_class_calls_remain_unchanged(s1_case):
    case = s1_case
    before = _snapshot(case["root"])
    resolver = S1GRiTSResolver(case["root"])
    pair = _pair(resolver, case)
    values, valid, months = S1GRiTSResolver.read_monthly(pair, 2, 2, patch_size=3)
    static, static_valid = S1GRiTSResolver.read_static(pair, 2, 2, patch_size=3)
    assert values.dtype == np.float32 and values.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(values[0], np.ones((2, 3, 3)))
    np.testing.assert_array_equal(values[1], np.full((2, 3, 3), -2))
    assert valid.all() and static_valid.all() and static.shape == (6, 3, 3)
    assert months.tolist() == ["2025-01", "2025-02"]
    with pytest.raises(CatalogConsistencyError, match="[Pp]rovenance|metadata|[Ee]vidence"):
        S1GRiTSResolver(case["root"], require_provenance=True)
    assert _snapshot(case["root"]) == before


@pytest.mark.parametrize("kind", ["monthly", "static"])
def test_existing_pair_stops_when_pending_appears_before_read(s1_case, monkeypatch, kind):
    case = s1_case
    resolver = S1GRiTSResolver(case["root"])
    pair = _pair(resolver, case)
    _pending(case["root"])
    monkeypatch.setattr(s1_module, "open_dataset", lambda *args, **kwargs: pytest.fail("Pending read opened imagery"))
    with pytest.raises(CatalogConsistencyError, match="pending|unfinished|[Ww]rit"):
        getattr(S1GRiTSResolver, "read_" + kind)(pair, 2, 2, patch_size=3)


@pytest.mark.parametrize("kind", ["monthly", "static"])
def test_pending_during_materialized_read_discards_result(s1_case, monkeypatch, kind):
    case = s1_case
    resolver = S1GRiTSResolver(case["root"])
    pair = _pair(resolver, case)
    original = s1_module.open_dataset
    calls = []

    def interrupted_open(entry, **kwargs):
        dataset = original(entry, **kwargs)
        calls.append(entry)
        _pending(case["root"])
        return dataset

    monkeypatch.setattr(s1_module, "open_dataset", interrupted_open)
    with pytest.raises(CatalogConsistencyError, match="pending|unfinished|[Ww]rit"):
        getattr(S1GRiTSResolver, "read_" + kind)(pair, 2, 2, patch_size=3)
    assert len(calls) == 1


def test_absolute_catalog_relocation_keeps_exact_static_reference_and_bytes(s1_case, tmp_path):
    case, original = s1_case, s1_case["root"]
    rows = case["rows"]
    for row in rows:
        row["zarr_path"] = str(original / row["zarr_path"])
        if row["reference_zarr_path"]:
            row["reference_zarr_path"] = str(original / row["reference_zarr_path"])
    pq.write_table(pa.Table.from_pylist(rows), original / "catalog.parquet")
    expected = S1GRiTSResolver.read_monthly(_pair(S1GRiTSResolver(original), case), 2, 2, patch_size=3)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    before = _snapshot(relocated)
    # Remove the old pathname so a silently ignored mapping cannot pass.
    assert original.resolve().parent == tmp_path.resolve()
    original.rename(tmp_path / "old-source-unavailable")
    resolver = S1GRiTSResolver(relocated, root_map={str(original): str(relocated)})
    pair = _pair(resolver, case)
    assert pair.monthly.zarr_path == relocated / case["monthly"]
    assert pair.static.zarr_path == relocated / case["static"]
    actual = resolver.read_monthly(pair, 2, 2, patch_size=3)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    assert _snapshot(relocated) == before


@pytest.fixture
def multi_case(s1_case, tmp_path):
    root = tmp_path / "s2"
    store = root / s1_case["tile"] / "optical.zarr"
    store.parent.mkdir(parents=True)
    values = np.arange(25, dtype="uint16").reshape(1, 5, 5)
    group = zarr.open_group(str(store), mode="w", zarr_format=3).create_group("20m")
    group.create_array("red", data=values, dimension_names=["time", "y", "x"])
    group.create_array("time", data=np.asarray(["2025-01-15"], dtype="datetime64[ns]"), dimension_names=["time"])
    group.create_array("x", data=np.arange(5) * 30 + 15, dimension_names=["x"])
    group.create_array("y", data=135 - np.arange(5) * 30, dimension_names=["y"])
    group.attrs.update(grid_id="grid")
    row = _row(s1_case["tile"], "scenes", f"{s1_case['tile']}/optical.zarr", 18, item="optical-1")
    row.update(zarr_group="20m", bands=json.dumps(["red"]), datetime="2025-01-15T00:00:00Z",
               source_collection="sentinel-2-c1-l2a")
    pq.write_table(pa.Table.from_pylist([row]), root / "catalog.parquet")
    return {"s1": s1_case, "root": root, "values": values[0]}


def test_multi_legacy_pixels_are_unchanged_and_pending_checked_before_read(multi_case, monkeypatch):
    case = multi_case
    resolver = MultiSensorResolver(case["s1"]["root"], case["root"])
    window = TargetWindow(0, 0, 5, 5)
    before = _snapshot(case["root"])
    result = resolver.read_s2_on_s1(case["s1"]["tile"], "red", s2_time_index=0, window=window)
    np.testing.assert_array_equal(result, case["values"])
    assert _snapshot(case["root"]) == before
    _pending(case["root"])
    monkeypatch.setattr(resolver_module, "read_aligned_variables", lambda *a, **kw: pytest.fail("Pending optical read reached raster IO"))
    with pytest.raises(CatalogConsistencyError, match="pending|unfinished|[Ww]rit"):
        resolver.read_s2_on_s1(case["s1"]["tile"], "red", s2_time_index=0, window=window)


def test_multi_discards_result_if_pending_appears_during_read(multi_case, monkeypatch):
    case = multi_case
    resolver = MultiSensorResolver(case["s1"]["root"], case["root"])
    original = resolver_module.read_aligned_variables
    calls = []

    def interrupted_read(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        _pending(case["root"])
        return result

    monkeypatch.setattr(resolver_module, "read_aligned_variables", interrupted_read)
    with pytest.raises(CatalogConsistencyError, match="pending|unfinished|[Ww]rit"):
        resolver.read_s2_on_s1(case["s1"]["tile"], "red", s2_time_index=0, window=TargetWindow(0, 0, 5, 5))
    assert calls == [True]


def test_time_join_detects_pending_started_during_coordinate_read(multi_case, monkeypatch):
    case = multi_case
    resolver = MultiSensorResolver(case["s1"]["root"], case["root"])
    original = resolver_module.open_dataset

    def interrupted_open(entry, **kwargs):
        result = original(entry, **kwargs)
        _pending(case["s1"]["root"])
        return result

    monkeypatch.setattr(resolver_module, "open_dataset", interrupted_open)
    with pytest.raises(CatalogConsistencyError, match="pending|unfinished|[Ww]rit"):
        resolver.time_join(case["s1"]["tile"])


def _monthly_outputs(case, generation=None):
    group = zarr.open_group(str(case["root"] / case["monthly"]), mode="a", use_consolidated=False)
    group.attrs.update(grid_id="grid")
    return [{"output_id": f"output-month-{index}", "generation_id": generation,
             "store_path": case["monthly"], "zarr_group": "", "variable_scope": ["VV_dB", "VH_dB", "n_obs"],
             "tile_id": case["tile"], "geometry_group_id": case["rows"][0]["geometry_group_id"],
             "grid_id": "grid", "product_type": "smonthly", "record_kind": "monthly",
             "time_index": index, "time_value_utc": when, "evidence_status": "artifact_observed",
             "unknown_reason": "Synthetic fixture; upstream scientific processing unknown"}
            for index, when in enumerate(["2025-01-15T00:00:00Z", "2025-02-15T00:00:00Z"])]


def test_same_index_generation_change_requires_new_reader_and_updates_sources(s1_case):
    case = s1_case
    rows = _monthly_outputs(case, "g1")
    source = {"source_id": "source-g1", "product_id": "input-v1", "evidence_status": "source_declared"}
    link = {"output_id": rows[0]["output_id"], "generation_id": "g1", "source_id": source["source_id"], "role": "used"}
    first = publish(case["root"], rows, sources=[source], source_links=[link])
    resolver = S1GRiTSResolver(case["root"], require_provenance=True)
    pair = _pair(resolver, case)
    before = resolver.read_monthly(pair, 2, 2, patch_size=3)
    old = resolver.explain_output(store_path=case["monthly"], time_index=0)
    assert old["outputs"][0]["output"]["generation_id"] == "g1"
    group = zarr.open_group(str(case["root"] / case["monthly"]), mode="a", use_consolidated=False)
    group["VV_dB"][0] = 7.0
    rows[0]["generation_id"] = "g2"
    source.update(source_id="source-g2", product_id="input-v2")
    link.update(generation_id="g2", source_id=source["source_id"])
    second = publish(case["root"], rows, sources=[source], source_links=[link])
    assert first["data_revision"] != second["data_revision"]
    with pytest.raises(CatalogConsistencyError, match="reload|changed"):
        resolver.read_monthly(pair, 2, 2, patch_size=3)
    with pytest.raises(CatalogConsistencyError, match="reload|changed"):
        resolver.explain_output(time_index=0)
    reloaded = S1GRiTSResolver(case["root"], require_provenance=True)
    current = reloaded.explain_output(store_path=case["monthly"], time_index=0)["outputs"][0]
    assert current["output"]["output_id"] == old["outputs"][0]["output"]["output_id"]
    assert current["output"]["time_index"] == 0 and current["output"]["generation_id"] == "g2"
    assert [entry["source"]["source_id"] for entry in current["sources"]] == ["source-g2"]
    after = reloaded.read_monthly(_pair(reloaded, case), 2, 2, patch_size=3)
    np.testing.assert_array_equal(after[0][0, 0], np.full((3, 3), 7))
    np.testing.assert_array_equal(after[0][:, 1], before[0][:, 1])
    assert reloaded.explain_output(time_index=1)["outputs"][0]["output"]["generation_id"] == "g1"


def test_detached_historical_metadata_and_pixels_survive_relocation(s1_case, tmp_path):
    case = s1_case
    rows = _monthly_outputs(case)
    metadata = tmp_path / "metadata"
    publish(metadata, rows)
    original = S1GRiTSResolver(case["root"], metadata_root=metadata, require_provenance=True)
    expected = original.read_monthly(_pair(original, case), 2, 2, patch_size=3)
    moved = tmp_path / "delivery"
    shutil.copytree(case["root"], moved / "data")
    shutil.copytree(metadata, moved / "metadata")
    before = _snapshot(moved)
    reloaded = S1GRiTSResolver(moved / "data", metadata_root=moved / "metadata", require_provenance=True)
    explained = reloaded.explain_output(store_path=case["monthly"], time_index=0)
    assert explained["metadata_status"] == "available"
    record = explained["outputs"][0]["output"]
    assert record["generation_id"] is None and record["unknown_reason"]
    assert record["time_value_utc"] == "2025-01-15T00:00:00Z"
    actual = reloaded.read_monthly(_pair(reloaded, case), 2, 2, patch_size=3)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert _snapshot(moved) == before


def _optical_outputs(case):
    return [{"output_id": "output-optical", "generation_id": "optical-g1",
             "store_path": f"{case['s1']['tile']}/optical.zarr", "zarr_group": "20m", "variable_scope": ["red"],
             "tile_id": case["s1"]["tile"], "grid_id": "grid", "product_type": "scenes", "record_kind": "scene",
             "source_item_id": "optical-1", "time_index": 0, "time_value_utc": "2025-01-15T00:00:00Z",
             "evidence_status": "artifact_observed", "unknown_reason": "Synthetic fixture; source scale is not applied"}]


def test_bound_s2_windows_reuse_loaded_identity_index(multi_case, monkeypatch):
    case = multi_case
    publish(case["root"], _optical_outputs(case))
    resolver = MultiSensorResolver(case["s1"]["root"], case["root"])
    monkeypatch.setattr(resolver_module, "group_scene_indices", lambda *a: pytest.fail("Window read rebuilt native scene identity index"))
    for _ in range(2):
        result = resolver.read_s2_on_s1(case["s1"]["tile"], "red", s2_time_index=0, window=TargetWindow(0, 0, 5, 5))
        np.testing.assert_array_equal(result, case["values"])
    _pending(case["root"])
    with pytest.raises(CatalogConsistencyError, match="pending|unfinished|[Ww]rit"):
        resolver.read_s2_on_s1(case["s1"]["tile"], "red", s2_time_index=0, window=TargetWindow(0, 0, 5, 5))


def test_explicit_external_target_grid_does_not_require_its_raster(s1_case, tmp_path):
    case = _quality_fixture.__wrapped__(tmp_path / "q")
    resolver = MultiSensorResolver(s1_case["root"], case["spectral"],
                                   s2_quality_root=case["quality"], s2_quality_pairing="current")
    pair = resolver.s2_scene_quality("17MNR", "scene_0")
    target = replace(pair.spectral_20m, sensor="sentinel-1", zarr_path=tmp_path / "never-created-target.zarr")
    result = resolver.read_s2_quality_on_s1(pair, window=TargetWindow(0, 0, 6, 6), target_entry=target)
    assert not target.zarr_path.exists()
    np.testing.assert_array_equal(result["scl"], case["values"]["scene_0"])
    assert result["scl"].dtype == np.uint8 and not result["quality_valid"][0, 0]
