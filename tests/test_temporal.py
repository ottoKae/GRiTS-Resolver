import numpy as np

from grits_resolver.temporal import join_times
from grits_resolver.aliases import COMMON_OPTICAL_ALIASES, resolve_alias
from grits_resolver.spatial import resampling_for_variable
from rasterio.enums import Resampling


def test_explicit_temporal_join_modes_and_tolerance() -> None:
    left = np.array(["2024-01-10", "2024-02-20"], dtype="datetime64[D]")
    right = np.array(["2024-01-01", "2024-01-21", "2024-03-01"], dtype="datetime64[D]")
    nearest = join_times(left, right, method="nearest", tolerance=np.timedelta64(15, "D"))
    assert [match.right_index for match in nearest] == [0, 2]
    backward = join_times(left, right, method="backward")
    assert [match.right_index for match in backward] == [0, 1]
    exact = join_times(left, right, method="exact")
    assert [match.right_index for match in exact] == [None, None]
    same = join_times(right[:1], right, method="backward")
    assert same[0].right_index == 0


def test_common_aliases_do_not_confuse_red_edge_and_nir() -> None:
    assert COMMON_OPTICAL_ALIASES == (
        "blue", "green", "red", "nir_narrow", "swir_1", "swir_2"
    )
    assert resolve_alias("sentinel-2", "nir_narrow") == "nir_narrow"
    assert resolve_alias("hls-s30", "nir_narrow") == "B8A"
    assert resampling_for_variable("red") == Resampling.bilinear
    assert resampling_for_variable("scl") == Resampling.nearest
    assert resampling_for_variable("cloud_probability") == Resampling.nearest
