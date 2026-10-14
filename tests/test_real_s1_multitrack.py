from pathlib import Path
import os

import numpy as np
import pytest

from grits_resolver import S1GRiTSResolver


pytestmark = pytest.mark.skipif(
    not os.environ.get("GRITS_TEST_S1_ROOT"),
    reason="Set GRITS_TEST_S1_ROOT for the real Ecuador 11-tile fixture",
)
ROOT = Path(os.environ.get("GRITS_TEST_S1_ROOT", "."))


def test_real_catalog_preserves_tracks_and_pairs_static() -> None:
    assert (ROOT / "catalog.parquet").exists()
    resolver = S1GRiTSResolver(ROOT)
    expected = {
        "17MNT": 2, "17MNU": 2, "17MNV": 2, "17MPT": 1,
        "17MPU": 1, "17MPV": 1, "17MQV": 2, "17NPA": 1,
        "17NQA": 2, "17NQB": 2, "17NRA": 1,
    }
    assert {tile: len(resolver.tracks(tile, require_static=True)) for tile in resolver.tiles()} == expected
    for tile in resolver.tiles():
        for pair in resolver.tracks(tile, require_static=True):
            assert pair.static is not None
            assert pair.static.row["reference_zarr_path"] == pair.monthly.row["zarr_path"]
            assert pair.static.grid_id == pair.monthly.grid_id


def test_real_track_read_is_not_arbitrated() -> None:
    assert (ROOT / "catalog.parquet").exists()
    resolver = S1GRiTSResolver(ROOT)
    pairs = resolver.tracks("17MNT", require_static=True)
    assert len(pairs) == 2
    # Find a common static center using a small strided read, then read each
    # monthly store independently at the exact same grid coordinate.
    datasets = []
    try:
        from grits_resolver.spatial import open_dataset
        for pair in pairs:
            datasets.append(open_dataset(pair.static))
        step = 64
        masks = [np.isfinite(np.asarray(ds.local_inc_angle.values[::step, ::step])) for ds in datasets]
        positions = np.argwhere(masks[0] & masks[1])
        assert len(positions)
        row, col = (positions[len(positions) // 2] * step).tolist()
    finally:
        for dataset in datasets:
            dataset.close()
    for pair in pairs:
        sar, valid, months = resolver.read_monthly(pair, row, col, patch_size=9)
        static, static_valid = resolver.read_static(pair, row, col, patch_size=9)
        assert sar.shape[0] == 2 and sar.shape[2:] == (9, 9)
        assert valid.shape == sar.shape[1:]
        assert len(months) == sar.shape[1]
        assert static.shape == (6, 9, 9)
        assert static_valid.shape == (9, 9)
    common = resolver.read_common_pair("17MNT", row, col, patch_size=9)
    assert len(common["tracks"]) == 2
    assert common["sar"][0].shape == common["sar"][1].shape
    assert common["sar"][0].shape[1] == len(common["months"])
    assert common["common_valid"].shape == common["sar"][0].shape[1:]
    for sar in common["sar"]:
        assert np.all(sar[:, ~common["common_valid"]] == 0)
