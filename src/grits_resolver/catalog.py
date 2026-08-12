from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq


Sensor = Literal["sentinel-1", "sentinel-2"]


class CatalogConsistencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogEntry:
    sensor: Sensor
    tile_id: str
    product_type: str
    zarr_path: Path
    zarr_group: str | None
    grid_id: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    resolution: tuple[float, float]
    bands: tuple[str, ...]
    row: dict[str, Any]

    @property
    def track_key(self) -> str:
        """Stable catalog identity shared by a monthly store and its static store."""
        return str(self.row.get("geometry_group_id") or self.row.get("item_id") or self.zarr_path.stem)


def _bands(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parsed = json.loads(value)
        return tuple(str(v) for v in parsed)
    return tuple(str(v) for v in value)


def resolve_asset_path(dataset_root: Path, tile_id: str, stored_path: str) -> Path:
    """Resolve canonical global paths and legacy S1 tile-relative paths."""
    path = Path(stored_path)
    if path.is_absolute():
        return path.resolve()
    global_candidate = dataset_root / path
    tile_candidate = dataset_root / tile_id / path
    if global_candidate.exists():
        return global_candidate.resolve()
    if tile_candidate.exists():
        return tile_candidate.resolve()
    # Canonical v7 requires dataset-root relative paths; retain that deterministic
    # interpretation for missing assets so diagnostics name the correct contract.
    return global_candidate.resolve()


def _read_atomic_catalog(path: Path, retries: int = 3) -> list[dict[str, Any]]:
    """Read one committed Parquet generation; retry if it is atomically replaced."""
    if path.name.endswith(".tmp"):
        raise CatalogConsistencyError(f"Refusing transient catalog file: {path}")
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            before = path.stat()
            rows = pq.read_table(path).to_pylist()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                return rows
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)
    raise CatalogConsistencyError(f"Catalog changed or was unreadable during read: {path}") from last_error


def load_catalog(dataset_root: str | Path, sensor: Sensor) -> list[CatalogEntry]:
    root = Path(dataset_root).resolve()
    global_catalog = root / "catalog.parquet"
    if global_catalog.exists():
        rows = _read_atomic_catalog(global_catalog)
    else:
        rows = []
        for tile_catalog in root.glob("*/catalog.parquet"):
            rows.extend(_read_atomic_catalog(tile_catalog))
    entries: list[CatalogEntry] = []
    for row in rows:
        status = str(row.get("status") or "").lower()
        if status != "complete":
            raise CatalogConsistencyError(
                f"Catalog contains non-complete product: tile={row.get('tile_id')} "
                f"item={row.get('item_id')} status={status or '<missing>'}"
            )
        zarr_value = row.get("zarr_path")
        if not zarr_value:
            continue
        tile_id = str(row["tile_id"])
        transform = tuple(float(v) for v in row["transform"][:6])
        entries.append(
            CatalogEntry(
                sensor=sensor,
                tile_id=tile_id,
                product_type=str(row.get("product_type") or ""),
                zarr_path=resolve_asset_path(root, tile_id, str(zarr_value)),
                zarr_group=(str(row["zarr_group"]) if row.get("zarr_group") else None),
                grid_id=str(row["grid_id"]),
                crs=str(row["crs"]),
                transform=transform,
                width=int(row["width"]),
                height=int(row["height"]),
                resolution=(float(row["resolution_x"]), float(row["resolution_y"])),
                bands=_bands(row.get("bands")),
                row=row,
            )
        )
    return entries
