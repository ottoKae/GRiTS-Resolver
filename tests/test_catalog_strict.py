import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grits_resolver.catalog import CatalogConsistencyError, load_catalog, map_root_path, resolve_asset_path


def test_rejects_non_complete_catalog_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pq.write_table(
            pa.Table.from_pylist([
                {"tile_id": "17MPU", "item_id": "pending", "status": "processing", "zarr_path": "x.zarr"}
            ]),
            root / "catalog.parquet",
        )
        with pytest.raises(CatalogConsistencyError, match="non-complete"):
            load_catalog(root, "sentinel-2")


def test_root_map_matches_components_and_prefers_specific_root(tmp_path):
    old, new, special = tmp_path / "old", tmp_path / "new", tmp_path / "special"
    roots = {old: new, old / "tile": special}
    assert map_root_path(old / "tile" / "a.zarr", roots) == special / "a.zarr"
    assert map_root_path(old / "other.zarr", roots) == new / "other.zarr"
    assert map_root_path(tmp_path / "old_suffix" / "a.zarr", roots) == tmp_path / "old_suffix" / "a.zarr"


@pytest.mark.parametrize("roots", [{"relative": "D:/new"}, {"D:/old": "relative"}])
def test_root_map_requires_explicit_absolute_roots(tmp_path, roots):
    with pytest.raises(ValueError, match="absolute"):
        map_root_path(tmp_path / "data.zarr", roots)


@pytest.mark.parametrize("stored", ["../outside.zarr", "unmapped_absolute"])
def test_mapped_catalog_must_resolve_inside_current_root(tmp_path, stored):
    root = tmp_path / "current"
    root.mkdir()
    value = str(tmp_path / "outside.zarr") if stored == "unmapped_absolute" else stored
    with pytest.raises(CatalogConsistencyError, match="dataset root"):
        resolve_asset_path(root, "tile", value, root_map={tmp_path / "old": root})
