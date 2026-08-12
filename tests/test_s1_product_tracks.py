import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from grits_resolver import S1GRiTSResolver


def _write_dynamic(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((2, 5, 5), dtype="float32")
    xr.Dataset(
        {"VV_dB": (("time", "y", "x"), data),
         "VH_dB": (("time", "y", "x"), data - 3),
         "n_obs": (("time", "y", "x"), np.ones_like(data, dtype="uint16"))},
        coords={"time": np.array(["2025-01-15", "2025-02-15"], dtype="datetime64[ns]"),
                "y": np.arange(5), "x": np.arange(5)},
    ).to_zarr(path, mode="w", consolidated=False)


def _write_static(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ones((5, 5), dtype="float32")
    xr.Dataset({name: (("y", "x"), values.copy()) for name in (
        "local_inc_angle", "inc_angle", "ls_map", "number_of_looks",
        "rtc_anf_beta0", "rtc_anf_sigma0",
    )}, coords={"y": np.arange(5), "x": np.arange(5)}).to_zarr(
        path, mode="w", consolidated=False,
    )


def _row(tile: str, product: str, rel: str, track: int, *, item: str,
         reference: str | None = None) -> dict:
    return {
        "item_id": item, "status": "complete", "tile_id": tile,
        "product_type": product, "zarr_path": rel, "zarr_group": None,
        "grid_id": "grid", "crs": "EPSG:32717",
        "transform": [30.0, 0.0, 0.0, 0.0, -30.0, 150.0],
        "width": 5, "height": 5, "resolution_x": 30.0, "resolution_y": 30.0,
        "bands": json.dumps(["VV_dB", "VH_dB", "n_obs"] if product != "static" else [
            "local_inc_angle", "inc_angle", "ls_map", "number_of_looks",
            "rtc_anf_beta0", "rtc_anf_sigma0"]),
        "geometry_group_id": f"{tile}_ASCENDING_TK{track}", "track": track,
        "reference_zarr_path": reference,
    }


def test_scenes_smonthly_and_static_remain_separate_by_track(tmp_path: Path) -> None:
    tile = "17AAA"
    rows = []
    for track in (18, 91):
        monthly = f"smonthly_ASCENDING/zarr/monthly_TK{track}.zarr"
        scenes = f"scenes_ASCENDING/zarr/scenes_TK{track}.zarr"
        static = f"static_ASCENDING/zarr/static_TK{track}.zarr"
        _write_dynamic(tmp_path / tile / monthly)
        _write_dynamic(tmp_path / tile / scenes)
        _write_static(tmp_path / tile / static)
        rows.extend([
            _row(tile, "smonthly", monthly, track, item=f"monthly-{track}"),
            _row(tile, "scenes", scenes, track, item=f"scene-{track}-a"),
            # Acquisition-level duplicate catalog row for the same scenes store.
            _row(tile, "scenes", scenes, track, item=f"scene-{track}-b"),
            _row(tile, "static", static, track, item=f"static-{track}", reference=monthly),
        ])
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "catalog.parquet")

    resolver = S1GRiTSResolver(tmp_path)
    assert len(resolver.product_tracks(tile, "smonthly")) == 2
    assert len(resolver.product_tracks(tile, "scenes")) == 2
    assert len(resolver.product_tracks(tile, "static")) == 2

    monthly_pairs = resolver.tracks(tile, product_type="smonthly", require_static=True)
    scenes_pairs = resolver.tracks(tile, product_type="scenes", require_static=True)
    assert [pair.static_match for pair in monthly_pairs] == ["reference", "reference"]
    assert [pair.static_match for pair in scenes_pairs] == ["geometry_group", "geometry_group"]
    assert [pair.track_key for pair in monthly_pairs] == [pair.track_key for pair in scenes_pairs]
