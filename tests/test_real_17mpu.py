from pathlib import Path

import numpy as np

from grits_resolver import MultiSensorResolver
from grits_resolver.spatial import TargetWindow


S1 = Path(r"D:\QGIS\s1grits-dataset\Ecuador_Datasets\Dong_11tiles_nospeckel_smonthly_blockwise_onlyTK")
S2 = Path(r"D:\QGIS\s1grits-dataset\Ecuador_Datasets\S2-GRiTS-17MPU-2024-test")


def test_real_catalog_paths_time_and_lazy_spatial_window() -> None:
    if not (S1 / "catalog.parquet").exists() or not any(S2.glob("*/catalog.parquet")):
        return
    resolver = MultiSensorResolver(S1, S2)
    s1 = resolver.s1_entry("17MPU")
    s2 = resolver.s2_entry("17MPU", "20m")
    assert s1.zarr_path.exists()
    assert s2.zarr_path.exists()
    assert s1.resolution == (30.0, 30.0)
    assert s2.resolution == (20.0, 20.0)
    assert resolver.spatial_alignment_mode(s2, s1) == "virtual_reprojection"
    matches = resolver.time_join(
        "17MPU", method="nearest", tolerance=np.timedelta64(20, "D")
    )
    assert len(matches) == 4
    assert all(match.right_index is not None for match in matches)
    data = resolver.read_s2_on_s1(
        "17MPU", "nir_narrow", s2_time_index=0,
        window=TargetWindow(row_off=1300, col_off=2200, height=256, width=256),
    )
    assert data.shape == (256, 256)
    assert np.isfinite(data).any()
    batch = resolver.read_s2_variables_on_s1(
        "17MPU", ["nir_narrow", "swir_1", "scl", "cloud_probability"],
        s2_time_index=0,
        window=TargetWindow(row_off=1300, col_off=2200, height=128, width=128),
    )
    assert set(batch) == {"nir_narrow", "swir_1", "scl", "cloud_probability"}
    assert batch["nir_narrow"].dtype == np.float32
    assert batch["scl"].dtype == np.uint8
