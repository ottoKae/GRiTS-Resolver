# GRiTS-Resolver

```text
S1-GRiTS ── produces ──┐
                       ├── catalog + Zarr ──> GRiTS-Resolver ──> applications / models
S2-GRiTS ── produces ──┘
```

S1-GRiTS and S2-GRiTS are data producers. GRiTS-Resolver is the independent,
read-only consumer layer that discovers product instances, preserves sensor
and track identity, pairs related products, joins calendars, and performs
windowed spatial alignment for downstream applications.

Read-only resolvers for S1-GRiTS and S2-GRiTS canonical Parquet catalogs and
Zarr stores. Data production remains the responsibility of S1-GRiTS and
S2-GRiTS.

## Installation

```powershell
conda env create -f environment.yml
conda activate py312_grits_resolver
python -m pip install -e . --no-build-isolation
```

## S1 track semantics

An MGRS tile may contain multiple Sentinel-1 tracks for `scenes`, `smonthly`,
and `static`. The resolver never silently collapses them. Product instances
are deduplicated by Zarr store, while their track identity is preserved by the
catalogue `geometry_group_id`.

```python
monthly = resolver.product_tracks("17MNT", "smonthly")
scenes = resolver.product_tracks("17MNT", "scenes")
static = resolver.product_tracks("17MNT", "static")
```

For a dynamic product, `tracks()` returns every track together with the
geometry-correct RTC-STATIC companion. Pairing first uses the exact
`reference_zarr_path`; when one static product is shared by `scenes` and
`smonthly`, the other dynamic product is paired using the common
`geometry_group_id` and `grid_id`.

```python
from grits_resolver import S1GRiTSResolver

root = r"D:\QGIS\s1grits-dataset\Ecuador_Datasets\Dong_11tiles_nospeckel_smonthly_blockwise_onlyTK"
resolver = S1GRiTSResolver(root)
pairs = resolver.tracks("17MNT", product_type="smonthly", require_static=True)

assert len(pairs) == 2                 # TK18 and TK91
for pair in pairs:
    print(pair.monthly.zarr_path)
    print(pair.static.zarr_path)
```

Scene stores use the same non-collapsing rule:

```python
scene_pairs = resolver.tracks(
    "17MNT", product_type="scenes", require_static=True,
)
```

For same-coordinate cross-track diagnostics, read every pair independently.
Because the monthly and static products share `grid_id`, transform and shape,
the same row/column denotes the same ground grid cell:

```python
row, col = 2000, 3000
observations = []
for pair in pairs:
    sar, valid, months = resolver.read_monthly(pair, row, col, patch_size=9)
    static, static_valid = resolver.read_static(pair, row, col, patch_size=9)
    observations.append((sar, valid, months, static, static_valid))

common = resolver.common_months(observations[0][2], observations[1][2])
```

The complete paired operation is available directly:

```python
paired = resolver.read_common_pair("17MNT", row=2000, col=3000, patch_size=9)
# paired["sar"]          -> two [2, common_time, 9, 9] arrays
# paired["common_valid"] -> one shared [common_time, 9, 9] mask
# paired["static"]       -> two [6, 9, 9] arrays
```

`sar` has shape `[2,time,9,9]`, `valid` has shape `[time,9,9]`, and
`static` has shape `[6,9,9]` in this order:

1. `local_inc_angle`
2. `inc_angle`
3. `ls_map`
4. `number_of_looks`
5. `rtc_anf_beta0`
6. `rtc_anf_sigma0`

The resolver does not combine months from different tracks. A downstream
best-track workflow may explicitly rank complete tracks, while a causal
cross-track experiment must retain all tracks and impose its own common-valid
calendar mask.

## S1/S2 virtual cube

`MultiSensorResolver` aligns selected S2 windows to an S1 30 m target grid and
supports explicit `exact`, `nearest`, `backward`, and `forward` time joins.

```python
import numpy as np
from grits_resolver import MultiSensorResolver
from grits_resolver.spatial import TargetWindow

resolver = MultiSensorResolver(s1_root, s2_root)
matches = resolver.time_join(
    "17MPU", method="nearest", tolerance=np.timedelta64(20, "D")
)
nir = resolver.read_s2_on_s1(
    "17MPU", "nir_narrow", s2_time_index=0,
    window=TargetWindow(1300, 2200, 256, 256),
)
```

All operations are lazy or windowed and do not modify source catalogs or Zarr
stores.
