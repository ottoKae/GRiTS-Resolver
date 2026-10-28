# Changelog

## 0.3.1rc1 — 2026-10-28 (candidate)

- Read provenance release v1/v2 and explain exact output/generation evidence.
- Guard S1/S2 reads against pending writes, changed releases and stale Catalogs;
  validate independent SCL pairing and explicit root mappings after relocation.
- Preserve scene identity and raw optical DN; retain valid zero cloud probability
  inside SCL coverage and consistent point/window reprojection.
- Historical evidence stays historical. Readers never infer missing production
  steps or apply numeric corrections merely because metadata describes them.
- Candidate delivery only; concurrent production/reading and full-year
  incremental producer performance are not accepted workloads.

## 0.1.0 — 2026-08-11

- Added strict S1-GRiTS and S2-GRiTS canonical catalog adapters.
- Added legacy S1 tile-relative path normalization.
- Added explicit exact/nearest/backward/forward temporal joins.
- Added lazy S2-to-S1 window reprojection.
- Added single-open batch variable reads.
- Added bilinear reflectance and nearest categorical/mask policies.
- Added catalog atomic-generation checks and catalog/Zarr time validation.
- Added automatic GDAL_DATA discovery for Python 3.12 conda environments.
# 0.2.0 - 2026-08-12

- Preserve all Sentinel-1 track stores for `scenes`, `smonthly`, and `static`.
- Pair dynamic tracks with RTC-STATIC by exact reference or shared track-level
  geometry identity without best-track arbitration.
- Add same-coordinate, common-calendar paired-track reads for causal sensor
  diagnostics.
- Document the S1-GRiTS/S2-GRiTS producer and GRiTS-Resolver consumer boundary.
