"""Explicit Sentinel-2 acquisition identities and duplicate-safe mosaics.

Tile timestamps are not acquisition identities. This module deliberately requires
the original STAC datatake metadata; it never guesses from a calendar date.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
import numpy as np


def acquisition_key(properties: Mapping) -> str:
    """Satellite + datatake start + absolute orbit, excluding processing baseline."""
    raw = str(properties.get("s2:datatake_id", ""))
    match = re.fullmatch(r"G(S2[A-Z])_(\d{8}T\d{6})_(\d{6})_N\d{2}\.\d{2}", raw)
    if not match:
        raise ValueError(f"Missing or unsupported Sentinel-2 datatake identity: {raw!r}")
    satellite, start, orbit = match.groups()
    platform = properties.get("platform")
    if platform is not None and str(platform).lower() != "sentinel-2" + satellite[-1].lower():
        raise ValueError("Platform and datatake identity disagree")
    return f"{satellite}_{start}_{orbit}"


def first_valid_mosaic(observations: Iterable[tuple[np.ndarray, np.ndarray]]):
    """One value per pixel per acquisition, in caller-declared priority order.

    Each value is [pixel, band], with an all-band validity mask [pixel]. Invalid
    pixels remain NaN. The returned count is source multiplicity, NOT temporal
    sample weight. All bands come from the same selected tile observation.
    """
    output = selected = multiplicity = None
    for index, (values, valid) in enumerate(observations):
        values = np.asarray(values)
        valid = np.asarray(valid, dtype=bool)
        if values.ndim != 2 or valid.shape != values.shape[:1]:
            raise ValueError("Expected [pixel, band] values and [pixel] validity")
        if output is None:
            output = np.full(values.shape, np.nan, dtype=np.result_type(values.dtype, np.float32))
            selected = np.full(len(values), -1, dtype=np.int32)
            multiplicity = np.zeros(len(values), dtype=np.uint32)
        if values.shape != output.shape or not np.isfinite(values[valid]).all():
            raise ValueError("Inconsistent shape or nonfinite valid observation")
        take = valid & (selected < 0)
        output[take] = values[take]
        selected[take] = index
        multiplicity += valid
    if output is None:
        raise ValueError("An acquisition mosaic needs at least one observation")
    return output, selected, multiplicity
