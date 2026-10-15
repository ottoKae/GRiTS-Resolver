import numpy as np
import pytest
from grits_resolver.s2_acquisition import acquisition_key, first_valid_mosaic


def test_identity_ignores_tile_timestamp_and_processing_version():
    p = {"platform": "sentinel-2a", "s2:datatake_id": "GS2A_20200102T153621_023456_N05.00"}
    q = dict(p, datetime="2020-01-02T15:40:00Z")
    q["s2:datatake_id"] = "GS2A_20200102T153621_023456_N05.09"
    assert acquisition_key(p) == acquisition_key(q)
    q["s2:datatake_id"] = "GS2A_20200102T170000_023457_N05.09"
    assert acquisition_key(p) != acquisition_key(q)
    with pytest.raises(ValueError): acquisition_key({"datetime": "2020-01-02"})
    with pytest.raises(ValueError): acquisition_key(dict(p, platform="sentinel-2b"))


def test_overlap_count_does_not_become_temporal_weight():
    a = np.array([[1., 2.], [np.nan, np.nan], [np.nan, np.nan]])
    b = np.array([[9., 9.], [3., 4.], [np.nan, np.nan]])
    value, selected, count = first_valid_mosaic([(a, [1, 0, 0]), (b, [1, 1, 0])])
    np.testing.assert_equal(value, [[1., 2.], [3., 4.], [np.nan, np.nan]])
    np.testing.assert_array_equal(selected, [0, 1, -1])
    np.testing.assert_array_equal(count, [2, 1, 0])
    # Adding a duplicate does not change the unique acquisition's contribution.
    repeat, _, _ = first_valid_mosaic([(a, [1, 0, 0]), (b, [1, 1, 0]), (b, [1, 1, 0])])
    np.testing.assert_equal(repeat, value)
    # Two genuine acquisitions, rather than three overlapping tile values.
    np.testing.assert_array_equal(np.median([value[0], [5., 6.]], axis=0), [3., 4.])


def test_mosaic_rejects_nonfinite_accepted_and_misaligned_shapes():
    with pytest.raises(ValueError): first_valid_mosaic([(np.array([[np.nan]]), [True])])
    with pytest.raises(ValueError): first_valid_mosaic([])
    with pytest.raises(ValueError): first_valid_mosaic([(np.ones((2, 1)), [True])])
