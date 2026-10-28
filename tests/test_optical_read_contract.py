"""Regression cases for scene identity and window-independent optical reads."""
from dataclasses import replace
import json
from types import SimpleNamespace

from affine import Affine
import numpy as np
import pytest
from rasterio.enums import Resampling
from rasterio.warp import reproject
import zarr

from grits_resolver.catalog import CatalogConsistencyError, CatalogEntry
from grits_resolver.resolver import MultiSensorResolver
from grits_resolver.s1 import S1GRiTSResolver
from grits_resolver.s2_quality import QualityCompanion, SceneQualityPair, group_scene_indices
from grits_resolver.spatial import TargetWindow, read_aligned_variables, read_aligned_window


def store(tmp_path, *, same_utc=False):
    path = tmp_path / "native.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    times = np.asarray(["2024-01-01", "2024-01-01" if same_utc else "2024-01-02"], dtype="datetime64[ns]")
    root.create_array("time", data=times, dimension_names=["time"])
    group = root.create_group("20m")
    group.create_array("time", data=times, dimension_names=["time"])
    rng = np.random.default_rng(41)
    values = rng.integers(100, 10000, size=(2, 84, 84), dtype="uint16")
    red = group.create_array("red", data=values, dimension_names=["time", "y", "x"], fill_value=0)
    red.attrs.update({"_FillValue": 0, "scale_factor": .0001, "add_offset": -.1})
    scl = np.full((2, 84, 84), 4, dtype="uint8")
    scl[:, 0, 0] = 0
    for name, data in (("scl", scl), ("cloud_probability", np.zeros(scl.shape, dtype="uint8"))):
        array = group.create_array(name, data=data, dimension_names=["time", "y", "x"], fill_value=0)
        array.attrs.update({"nodata": 0, "_FillValue": 0})
    rows = [{"item_id": item, "datetime": str(times[index]), "collection": "test",
             "resolution_groups": {"20m": ["red", "scl", "cloud_probability"]}}
            for index, item in enumerate(("scene-a", "scene-b"))]
    (path / "s2grits_resume_state.json").write_text(json.dumps({"items": rows}), encoding="utf-8")
    base = CatalogEntry("sentinel-2", "49QHE", "scenes", path, "20m", "grid", "EPSG:32649",
                        (20., 0., 500000., 0., -20., 2500000.), 84, 84, (20., 20.),
                        ("red", "scl", "cloud_probability"), {})
    entries = [replace(base, row={"item_id": row["item_id"], "datetime": row["datetime"],
                                  "source_collection": "test", "track": 32 if i == 0 else 132})
               for i, row in enumerate(rows)]
    return entries, values


@pytest.mark.parametrize("ratio", [0.5, 1.5, 3.5, 7.])
@pytest.mark.parametrize("method", [Resampling.nearest, Resampling.bilinear, Resampling.cubic])
def test_point_and_subwindow_match_full_source_reference(tmp_path, ratio, method):
    entries, values = store(tmp_path)
    source = entries[0]
    size = int(84 / ratio)
    target = replace(source, transform=(20 * ratio, 0., 500000., 0., -20 * ratio, 2500000.),
                     width=size, height=size, resolution=(20 * ratio, 20 * ratio))
    full = read_aligned_window(source, target, "red", time_index=0,
                               window=TargetWindow(0, 0, size, size), resampling=method)
    reference = np.full((size, size), np.nan, dtype="float32")
    reproject(values[0], reference, src_transform=Affine(*source.transform), src_crs=source.crs,
              dst_transform=Affine(*target.transform), dst_crs=target.crs,
              src_nodata=0, dst_nodata=np.nan, resampling=method)
    np.testing.assert_allclose(full, reference, atol=.001, rtol=0)
    for row, col in ((0, 0), (size // 2, size // 2), (size - 1, size - 1)):
        point = read_aligned_window(source, target, "red", time_index=0,
                                    window=TargetWindow(row, col, 1, 1), resampling=method)
        np.testing.assert_allclose(point[0, 0], full[row, col], atol=.001, rtol=0)
    small = read_aligned_window(source, target, "red", time_index=0,
                                window=TargetWindow(2, 2, 5, 5), resampling=method)
    np.testing.assert_allclose(small, full[2:7, 2:7], atol=.001, rtol=0)


def test_raw_dn_and_valid_cloud_zero_do_not_depend_on_requested_variables(tmp_path):
    entries, values = store(tmp_path)
    source = entries[0]
    window = TargetWindow(0, 0, 4, 4)
    one = read_aligned_variables(source, source, ["red", "cloud_probability"], time_index=0, window=window)
    batch = read_aligned_variables(source, source, ["red", "cloud_probability", "scl"], time_index=0, window=window)
    np.testing.assert_array_equal(one["red"], values[0, :4, :4])
    np.testing.assert_array_equal(one["red"], batch["red"])
    np.testing.assert_array_equal(one["cloud_probability"], batch["cloud_probability"])
    assert batch["cloud_probability"][0, 0] == 255
    assert np.all(batch["cloud_probability"][1:] == 0)


def resolver(entries):
    result = MultiSensorResolver.__new__(MultiSensorResolver)
    result.s2_entries = entries
    result.s1_entries = []
    result.quality_companion = None
    return result


def test_duplicate_utc_is_identified_by_committed_item_and_index(tmp_path):
    entries, _ = store(tmp_path, same_utc=True)
    reader = resolver(list(reversed(entries)))
    reader.validate_catalogs()
    assert reader.s2_entry("49QHE", time_index=0).row["item_id"] == "scene-a"
    assert reader.s2_entry("49QHE", item_id="scene-b").row["track"] == 132
    assert group_scene_indices(entries[0])[1]["scene-b"][0] == 1
    with pytest.raises(CatalogConsistencyError, match="does not match"):
        reader.s2_entry("49QHE", item_id="scene-a", time_index=1)
    with pytest.raises(CatalogConsistencyError, match="Ambiguous S2 tracks"):
        reader.s2_entry("49QHE")
    reader.s2_entries.append(entries[0])
    with pytest.raises(CatalogConsistencyError, match="Duplicate S2 catalog"):
        reader.validate_catalogs()


def test_duplicate_utc_without_identity_witness_is_not_silently_selected(tmp_path):
    entries, _ = store(tmp_path, same_utc=True)
    sidecar = entries[0].zarr_path / "s2grits_resume_state.json"
    data = json.loads(sidecar.read_text())
    for row in data["items"]:
        row.pop("resolution_groups")
    sidecar.write_text(json.dumps(data))
    with pytest.raises(CatalogConsistencyError, match="group commit identities"):
        resolver(entries).s2_entry("49QHE", item_id="scene-b")


def test_store_and_target_track_selection_reject_ambiguity(tmp_path):
    entries, values = store(tmp_path)
    reader = resolver(entries)
    first = replace(entries[0], sensor="sentinel-1", product_type="smonthly", row={"track": 18})
    second = replace(first, zarr_path=tmp_path / "other.zarr", row={"track": 91})
    reader.s1_entries = [first, second]
    with pytest.raises(CatalogConsistencyError, match="Ambiguous S1"):
        reader.s1_entry("49QHE")
    assert reader.s1_entry("49QHE", track=91) == second
    assert reader.s1_entry("49QHE", store=first.zarr_path) == first
    actual = reader.read_s2_on_s1("49QHE", "red", s2_item_id="scene-b", s1_track=18,
                                   window=TargetWindow(0, 0, 3, 3))
    np.testing.assert_array_equal(actual, values[1, :3, :3])
    with pytest.raises(ValueError, match="explicitly"):
        reader.read_s2_on_s1("49QHE", "red", s2_track=32, s1_track=18,
                             window=TargetWindow(0, 0, 3, 3))
    reader.s2_entries.append(replace(entries[0], zarr_path=tmp_path / "different.zarr"))
    with pytest.raises(CatalogConsistencyError, match="Ambiguous S2"):
        reader.s2_entry("49QHE", time_index=0)
    assert reader.s2_entry("49QHE", store=entries[0].zarr_path, time_index=1) == entries[1]


def test_quality_legacy_index_rejects_multiple_stores():
    companion = QualityCompanion.__new__(QualityCompanion)
    companion.ensure_current = lambda: None
    companion._at = {("49QHE", "20m", 0): [("one", "49QHE", "a"), ("two", "49QHE", "b")]}
    with pytest.raises(CatalogConsistencyError, match="ambiguous"):
        companion.scene_at("49QHE", "20m", 0)


def test_cloud_coverage_uses_paired_quality_group_index(tmp_path):
    spectral, _ = store(tmp_path / "spectral")
    quality, _ = store(tmp_path / "quality")
    root = zarr.open_group(str(spectral[0].zarr_path), mode="a")
    del root["20m"]["scl"]
    qroot = zarr.open_group(str(quality[0].zarr_path), mode="a")
    qroot["20m"]["scl"][1, 1, 1] = 0
    pair = SceneQualityPair("test", "49QHE", "scene-a", spectral[0], 0,
                            spectral[0], 0, quality[0], 1)
    reader = resolver(spectral)
    reader.s1_entries = [replace(spectral[0], sensor="sentinel-1", product_type="smonthly")]
    reader.quality_companion = SimpleNamespace(ensure_current=lambda pair=None: None, scene=lambda *a, **kw: pair)
    out = reader.read_s2_variables_on_s1("49QHE", ["cloud_probability", "scl"],
        s2_item_id="scene-a", window=TargetWindow(0, 0, 3, 3))
    assert out["cloud_probability"][1, 1] == 255
    assert out["cloud_probability"][2, 2] == 0
    np.testing.assert_array_equal(out["cloud_probability"] == 255, out["scl"] == 0)


def test_s1_no_common_months_returns_empty_time_dimension():
    reader = S1GRiTSResolver.__new__(S1GRiTSResolver)
    pairs = [SimpleNamespace(name=name, monthly=SimpleNamespace(zarr_path=name), static=None)
             for name in ("one", "two")]
    reader.tracks = lambda *args, **kwargs: pairs
    reader.provenance = SimpleNamespace(assert_current=lambda **kwargs: None)
    reader.read_monthly = lambda pair, *args, **kwargs: (
        np.ones((2, 1, 1, 1)), np.ones((1, 1, 1), bool),
        np.array(["2024-01" if pair.name == "one" else "2024-02"]))
    reader.read_static = lambda *args, **kwargs: (np.ones((6, 1, 1)), np.ones((1, 1), bool))
    actual = reader.read_common_pair("49QHE", 0, 0, patch_size=1)
    assert actual["months"].shape == (0,)
    assert actual["common_valid"].shape == (0, 1, 1)
    assert all(value.shape == (2, 0, 1, 1) for value in actual["sar"])
