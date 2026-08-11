from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np

from .resolver import MultiSensorResolver
from .spatial import TargetWindow, open_dataset


def benchmark_17mpu(s1_root: str | Path, s2_root: str | Path) -> dict:
    started = time.perf_counter()
    resolver = MultiSensorResolver(s1_root, s2_root)
    catalog_seconds = time.perf_counter() - started

    started = time.perf_counter()
    matches = resolver.time_join(
        "17MPU", method="nearest", tolerance=np.timedelta64(20, "D")
    )
    temporal_seconds = time.perf_counter() - started
    ds1 = open_dataset(resolver.s1_entry("17MPU"))
    ds2 = open_dataset(resolver.s2_entry("17MPU", "20m"))
    try:
        s1_time_count = int(ds1.sizes["time"])
        s2_time_count = int(ds2.sizes["time"])
    finally:
        ds1.close(); ds2.close()

    spatial: dict[str, dict] = {}
    for size in (128, 256, 512):
        tracemalloc.start()
        started = time.perf_counter()
        data = resolver.read_s2_on_s1(
            "17MPU", "nir_narrow", s2_time_index=0,
            window=TargetWindow(1300, 2200, size, size),
        )
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        spatial[str(size)] = {
            "seconds": elapsed,
            "peak_python_bytes": peak,
            "finite_fraction": float(np.isfinite(data).mean()),
        }
    return {
        "catalog_seconds": catalog_seconds,
        "temporal_seconds": temporal_seconds,
        "s1_time_count": s1_time_count,
        "s2_time_count": s2_time_count,
        "matched_count": sum(match.right_index is not None for match in matches),
        "spatial_windows": spatial,
    }
