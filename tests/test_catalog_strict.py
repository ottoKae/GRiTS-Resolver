import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grits_resolver.catalog import CatalogConsistencyError, load_catalog


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
