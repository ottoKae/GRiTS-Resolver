"""A neighbor's identity must never determine the requested target grid."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from grits_resolver.resolver import MultiSensorResolver
from grits_resolver.spatial import TargetWindow
from test_s2_quality import entries

def setup(entries):
    source,target,expected=entries
    source=replace(source,zarr_group='10m',row={'datetime':'2020-01-02T12:00:00Z',
        'source_collection':'test','item_id':'neighbor'})
    target=replace(target,tile_id='17MPU')
    pair=SimpleNamespace(tile_id=source.tile_id,item_id='neighbor',source_collection='test',
        quality_available=True,quality_20m=replace(source,zarr_group='20m'),quality_20m_index=0)
    calls=[]
    def scene(tile,item,**kwargs):
        calls.append((tile,item,kwargs));return pair
    r=MultiSensorResolver.__new__(MultiSensorResolver)
    r.quality_companion=SimpleNamespace(ensure_current=lambda:None,scene=scene)
    r.s1_entries=[target];r.s2_entries=[source]
    return r,pair,target,expected,calls

def test_neighbor_read_uses_target_and_preserves_source_identity(entries):
    r,pair,target,expected,calls=setup(entries)
    r.s1_entry=lambda tile: (_ for _ in ()).throw(AssertionError('Must use explicit target'))
    result=r.read_s2_quality_on_s1(pair,window=TargetWindow(0,0,6,6),target_entry=target)
    np.testing.assert_array_equal(result['scl'],expected)
    np.testing.assert_array_equal(result['quality_valid'],expected!=0)
    assert calls[0][0]=='17MNR'
    shifted=replace(target,transform=(20.,0.,500040.,0.,-20.,9600040.))
    other=r.read_s2_quality_on_s1(pair,window=TargetWindow(0,0,6,4),target_entry=shifted)
    np.testing.assert_array_equal(other['scl'],expected[:,2:])

def test_missing_quality_is_not_clear(entries):
    r,pair,target,_,_=setup(entries);pair.quality_available=False
    out=r.read_s2_quality_on_s1(pair,window=TargetWindow(0,0,2,3),target_entry=target)
    assert not out['quality_available'] and not out['quality_valid'].any()
    assert out['scl'].dtype==np.uint8 and out['scl'].shape==(2,3)

def test_partial_source_overlap_does_not_wrap_negative_row(entries):
    r,pair,target,_,_=setup(entries)
    # Target footprint overlaps the source, but its centre is 5 m north of it.
    target=replace(target,width=1,height=1,transform=(30.,0.,500000.,0.,-30.,9600060.))
    result=r.read_s2_quality_on_s1(pair,window=TargetWindow(0,0,1,1),target_entry=target)
    assert result['scl'][0,0]==0 and not result['quality_valid'][0,0]

def test_candidates_ignore_tile_names_but_respect_bounds_dates_and_dedup(entries):
    r,pair,target,_,calls=setup(entries);source=r.s2_entries[0]
    far=replace(source,transform=(20.,0.,700000.,0.,-20.,9600040.),row={**source.row,'item_id':'far'})
    late=replace(source,row={**source.row,'item_id':'late','datetime':'2020-02-01'})
    r.s2_entries += [source,replace(source,zarr_group='20m'),far,late]
    got=r.s2_quality_candidates_on_target(target,window=TargetWindow(0,0,6,6),start='2020-01-01',end='2020-02-01')
    assert got==[pair] and len(calls)==1 and calls[0][0]=='17MNR'
    with pytest.raises(ValueError,match='start < end'):
        r.s2_quality_candidates_on_target(target,window=TargetWindow(0,0,6,6),start='2020-02-01',end='2020-01-01')

def test_default_same_tile_read_still_works(entries):
    r,pair,target,expected,_=setup(entries)
    calls=[]
    def get(tile):calls.append(tile);return target
    r.s1_entry=get
    result=r.read_s2_quality_on_s1(pair,window=TargetWindow(0,0,6,6))
    np.testing.assert_array_equal(result['scl'],expected)
    assert calls==['17MNR']
