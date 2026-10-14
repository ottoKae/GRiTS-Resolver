"""Categorical reading tests use tiny local Zarrs, never production archives."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
from rasterio.enums import Resampling
import zarr

from grits_resolver.catalog import CatalogEntry
from grits_resolver.resolver import MultiSensorResolver
from grits_resolver.spatial import TargetWindow, open_dataset, read_aligned_variables


@pytest.fixture
def entries(tmp_path):
    path=tmp_path/"quality.zarr"
    root=zarr.open_group(str(path),mode="w",zarr_format=3)
    group=root.create_group("20m")
    group.create_array("time",data=np.asarray(["2023-01-01"],dtype="datetime64[ns]"),dimension_names=["time"])
    group.create_array("x",data=np.arange(6)*20+500010,dimension_names=["x"])
    group.create_array("y",data=9600030-np.arange(6)*20,dimension_names=["y"])
    data=np.asarray([[0,4,5,6,9,11]]*6,dtype="uint8")[None]
    array=group.create_array("scl",data=data,dimension_names=["time","y","x"],fill_value=0)
    array.attrs.update({"nodata":0,"_FillValue":0})
    transform=(20.,0.,500000.,0.,-20.,9600040.)
    source=CatalogEntry("sentinel-2","17MNR","scenes",path,"20m","test-grid","EPSG:32717",
                        transform,6,6,(20.,20.),("scl",),{})
    target=CatalogEntry("sentinel-1","17MNR","smonthly",tmp_path/"target.zarr",None,"test-grid",
                        "EPSG:32717",transform,6,6,(20.,20.),(),{})
    return source,target,data[0]


def test_fill_zero_and_dtype_preserved_on_locked_raw_xarray_path(entries):
    source,_,expected=entries
    dataset=open_dataset(source,raw=True)
    try:
        values=np.asarray(dataset.scl.isel(time=0))
        assert values.dtype==np.uint8
        np.testing.assert_array_equal(values,expected)
    finally:
        dataset.close()


def test_legacy_inline_scl_single_and_batch_nearest_and_read_only(entries):
    source,target,expected=entries
    resolver=MultiSensorResolver.__new__(MultiSensorResolver)
    resolver.s1_entries=[target]
    resolver.s2_entries=[source]
    resolver.quality_companion=None
    def snapshot():
        return {p.relative_to(source.zarr_path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.zarr_path.rglob("*") if p.is_file()}
    before=snapshot()
    window=TargetWindow(0,0,6,6)
    a=resolver.read_s2_on_s1("17MNR","scl",s2_time_index=0,window=window)
    b=resolver.read_s2_variables_on_s1("17MNR",["scl"],s2_time_index=0,window=window)["scl"]
    assert a.dtype==b.dtype==np.uint8
    np.testing.assert_array_equal(a,expected)
    np.testing.assert_array_equal(b,expected)
    assert before==snapshot()
    with pytest.raises(ValueError,match="nearest"):
        resolver.read_s2_on_s1("17MNR","scl",s2_time_index=0,window=window,resampling=Resampling.bilinear)


def test_outside_footprint_remains_uint8_zero_and_bilinear_never_allowed(entries):
    source,target,_=entries
    window=TargetWindow(1000,1000,4,3)
    result=read_aligned_variables(source,target,["scl"],time_index=0,window=window)["scl"]
    assert result.dtype==np.uint8 and result.shape==(4,3) and np.all(result==0)
    with pytest.raises(ValueError,match="nearest"):
        read_aligned_variables(source,target,["scl"],time_index=0,window=window,
                               resampling={"scl":Resampling.average})
