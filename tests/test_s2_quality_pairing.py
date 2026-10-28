"""Portable, local-only fixtures for the independent SCL pairing contract."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from grits_resolver.catalog import CatalogConsistencyError, load_catalog
from grits_resolver.provenance import canonical_json, data_revision
from grits_resolver.s2_quality import QualityCompanion, digest, public_receipt


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def file_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def pixel_digest(data):
    value = hashlib.sha256(json.dumps({"dtype": "uint8", "shape": list(data.shape), "block": 512},
                                     sort_keys=True).encode("ascii"))
    value.update(data.tobytes(order="C"))
    return value.hexdigest()


@pytest.fixture
def pair_case(tmp_path):
    tile = "17MNR"
    spectral, quality = tmp_path / "spectral", tmp_path / "quality"
    relative = f"{tile}/native.zarr"
    moments = [f"2023-0{i+1}-01T00:00:00Z" for i in range(3)]
    times = np.asarray([x.removesuffix("Z") for x in moments], dtype="datetime64[ns]")
    scene_ids = [f"scene_{i}" for i in range(3)]
    orders = {"10m": [2, 1, 0], "20m": [2, 0, 1]}
    rows = {"spectral": [], "quality": []}
    pixel_values = {}
    for kind, path in (("spectral", spectral), ("quality", quality)):
        store = zarr.open_group(str(path / relative), mode="w", zarr_format=3)
        store.create_array("time", data=times, dimension_names=["time"])
        for group_name in (["10m", "20m"] if kind == "spectral" else ["20m"]):
            group = store.create_group(group_name)
            scale = int(group_name[:-1])
            shape = 12 if scale == 10 else 6
            transform = [float(scale), 0., 500000., 0., -float(scale), 9600040.]
            grid_id = f"grid-{scale}"
            group.attrs.update(grid_id=grid_id, crs="EPSG:32717", width=shape, height=shape, transform=transform)
            order = orders[group_name] if kind == "spectral" else [0, 1, 2]
            group.create_array("time", data=times[order], dimension_names=["time"])
            group.create_array("x", data=500000. + (np.arange(shape)+.5)*scale, dimension_names=["x"])
            group.create_array("y", data=9600040. - (np.arange(shape)+.5)*scale, dimension_names=["y"])
            name = "scl" if kind == "quality" else ("red" if group_name == "10m" else "swir_1")
            values = (np.asarray([(np.arange(shape*shape).reshape(shape,shape)+i)%12 for i in order], dtype="uint8")
                      if kind == "quality" else np.asarray([np.full((shape,shape), 100+i) for i in order], dtype="uint16"))
            group.create_array(name, data=values, chunks=(1,shape,shape), dimension_names=["time","y","x"], fill_value=0)
            for i, scene_id in enumerate(scene_ids):
                rows[kind].append({"status":"complete", "tile_id":tile, "item_id":scene_id,
                    "source_collection":"sentinel-2-c1-l2a", "datetime":moments[i], "product_type":"scenes",
                    "product_variant":"quality_scl_v1" if kind=="quality" else "native",
                    "zarr_path":relative,"zarr_group":group_name,"grid_id":grid_id,"crs":"EPSG:32717",
                    "transform":transform,"width":shape,"height":shape,"resolution_x":float(scale),
                    "resolution_y":float(scale),"bands":[name]})
                if kind=="quality": pixel_values[scene_id]=values[i]
        write_json(path / relative / "s2grits_resume_state.json", {"items":[
            {"item_id":scene_ids[i],"datetime":moments[i],"collection":"sentinel-2-c1-l2a"} for i in range(3)]})
        pq.write_table(pa.Table.from_pylist(rows[kind]), path / "catalog.parquet")
    binding = {"spectral_root":str(spectral),"quality_root":str(quality),"cache_root":str(tmp_path/"cache"),
               "spectral_catalog_sha256":digest(spectral/"catalog.parquet")}
    links, evidence, outputs = [], {}, []
    witnesses = {relative+"/s2grits_resume_state.json":digest(quality/relative/"s2grits_resume_state.json")}
    for i, item in enumerate(scene_ids):
        link = {"source_collection":"sentinel-2-c1-l2a","tile_id":tile,"item_id":item,"datetime":moments[i],
                "source_sha256":str(i+1)*64}
        for prefix, group, index in (("spectral_10m","10m",orders["10m"].index(i)),
                                    ("spectral_20m","20m",orders["20m"].index(i)),("quality_20m","20m",i)):
            link.update({prefix+"_store":relative,prefix+"_group":group,prefix+"_time_index":index,
                         prefix+"_grid_id":"grid-"+group[:-1]})
        decoded = pixel_digest(pixel_values[item])
        receipt = {"status":"ACCEPTED","binding":binding,"item_id":item,"tile_id":tile,"source_sha256":link["source_sha256"],
                   "generation_id":"gen-"+item,"link":dict(link),"verification":{"full_pixel_comparison":True,
                   "compared_pixels":36,"mismatched_pixels":0,"decoded_sha256":decoded,"decoded_hash_scheme":"uint8-block512-v1"}}
        receipt_path = quality/f"_task_state/accepted/{tile}_{item}.json"
        write_json(receipt_path, receipt)
        evidence[tile+"_"+item]={"legacy_receipt_sha256":digest(receipt_path),"receipt":public_receipt(receipt),
                              "current_integrity":{"decoded_sha256":decoded,"integrity":"accepted_decoded_digest"}}
        logical={"collection":link["source_collection"],"tile_id":tile,"item_id":item,"group":"20m","grid_id":"grid-20","variables":["scl"]}
        outputs.append({"output_id":"output-"+hashlib.sha256(canonical_json(logical)).hexdigest(),"generation_id":"gen-"+item,
                        "store_path":relative,"zarr_group":"20m","variable_scope":["scl"],"tile_id":tile,"geometry_group_id":None,
                        "grid_id":"grid-20","product_type":"quality_scl","record_kind":"scene","time_index":i,
                        "time_value_utc":moments[i],"period_start":None,"period_end":None,"source_item_id":item})
        witnesses[f"decoded/{relative}/20m/scl/{item}"]=decoded
        links.append(link)
    pq.write_table(pa.Table.from_pylist(links),quality/"quality_links.parquet")
    artifacts={name:digest(quality/name) for name in ["catalog.parquet","quality_links.parquet",
               *[f"_task_state/accepted/{tile}_{item}.json" for item in scene_ids]]}
    marker={"state":"published","contract_version":1,"zarr_format":3,"consolidated":False,
            "environment":{n:importlib.metadata.version(n) for n in ("zarr","xarray","numcodecs")},"binding":binding,
            "artifacts":artifacts,"accepted_scenes":3,"accepted_prefixes":{tile:3},"generation":7}
    write_json(quality/"companion.json",marker)
    spectral_witness={"catalog.parquet":digest(spectral/"catalog.parquet"),relative+"/s2grits_resume_state.json":digest(spectral/relative/"s2grits_resume_state.json")}
    locations=[]
    for link in links:
        for group in ("10m","20m"):
            locations.append({"item_id":link["item_id"],**{name:link[f"spectral_{group}_{name}"] for name in ("store","group","time_index","grid_id")}})
    document={"contract_version":1,"run_id":"pair-v2","binding":{k:v for k,v in binding.items() if not k.endswith("_root")},
              "spectral_data_revision":"observed-"+hashlib.sha256(canonical_json({"witnesses":spectral_witness,"locations":locations})).hexdigest(),
              "spectral_revision_scope":"historical_structure_only","quality_data_revision":data_revision(outputs,witnesses),
              "quality_revision_scope":"historical_or_changed_observation","legacy_companion_sha256":digest(quality/"companion.json"),
              "resolver_support":"requires_P4_version_aware_pairing_reader"}
    for link in links:
        link.update(quality_generation_id="gen-"+link["item_id"],quality_available=True,unavailable_reason=None)

    def publish():
        directory=quality/"provenance/pairings/pair-v2"
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/"pairing.json",document)
        write_json(directory/"evidence.json",evidence)
        pq.write_table(pa.Table.from_pylist(links),directory/"quality_links.parquet")
        write_json(quality/"provenance/pairings/current.json",{"kind":"independent_quality_pairing","contract_version":1,
            "run_id":"pair-v2","request_sha256":hashlib.sha256(canonical_json({**document,"links":links,"evidence":evidence})).hexdigest(),
            "artifacts":{n:digest(directory/n) for n in ("pairing.json","evidence.json","quality_links.parquet")},
            "spectral_data_revision":document["spectral_data_revision"],"quality_data_revision":document["quality_data_revision"]})
    publish()
    return {"spectral":spectral,"quality":quality,"links":links,"document":document,"evidence":evidence,
            "publish":publish,"relative":relative,"values":pixel_values}


def reader(case, **kwargs):
    return QualityCompanion(case["spectral"],case["quality"],load_catalog(case["spectral"],"sentinel-2"),**kwargs)


def test_legacy_and_current_preserve_indices_values_and_files(pair_case):
    case=pair_case
    before={r:file_hashes(case[r]) for r in ("spectral","quality")}
    for mode in ("legacy","current"):
        value=reader(case,pairing=mode).scene("17MNR","scene_0")
        assert (value.spectral_10m_index,value.spectral_20m_index,value.quality_20m_index)==(2,1,0)
        data=zarr.open_group(str(value.quality_20m.zarr_path),mode="r",use_consolidated=False)["20m/scl"][value.quality_20m_index]
        np.testing.assert_array_equal(data,case["values"]["scene_0"])
        assert data.dtype==np.uint8 and data[0,0]==0
    assert before=={r:file_hashes(case[r]) for r in before}


def test_unavailable_pair_is_not_silently_reused(pair_case):
    case=pair_case
    case["links"][0].update(quality_available=False,unavailable_reason="source_scl_content_changed",source_sha256="f"*64)
    case["publish"]()
    result=reader(case,pairing="current")
    with pytest.raises(CatalogConsistencyError,match="No published quality"):
        result.scene("17MNR","scene_0")
    pair=result.scene("17MNR","scene_0",allow_missing=True)
    assert not pair.quality_available and pair.unavailable_reason=="source_scl_content_changed"


@pytest.mark.parametrize("target",["spectral","quality"])
def test_pending_blocks_loading_and_existing_reader(pair_case,target):
    case=pair_case
    result=reader(case,pairing="current")
    write_json(case[target]/"provenance/pending/write.json",{"phase":"writing"})
    with pytest.raises(CatalogConsistencyError,match="unfinished|pending"):
        reader(case,pairing="current")
    with pytest.raises(CatalogConsistencyError,match="unfinished|pending"):
        result.scene("17MNR","scene_0")


@pytest.mark.parametrize("change",["pairing","catalog","time"])
def test_cached_reader_rejects_changed_version_or_selected_store(pair_case,change):
    case=pair_case
    result=reader(case,pairing="current")
    pair=result.scene("17MNR","scene_0")
    if change=="time":
        root=zarr.open_group(str(case["spectral"]/case["relative"]),mode="a")
        root["10m/time"][2]=np.datetime64("2023-01-02","ns")
    else:
        path=case["quality"]/"provenance/pairings/current.json" if change=="pairing" else case["spectral"]/"catalog.parquet"
        path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(CatalogConsistencyError,match="changed"):
        result.ensure_current(pair)


def test_window_guards_do_not_rehash_catalogs_reopen_stores_or_read_pixels(pair_case,monkeypatch):
    import grits_resolver.s2_quality as module
    case=pair_case
    result=reader(case,pairing="current")
    pair=result.scene("17MNR","scene_0")
    monkeypatch.setattr(module,"digest",lambda *args:pytest.fail("Window guard rehashed files"))
    monkeypatch.setattr(module.zarr,"open_group",lambda *args,**kwargs:pytest.fail("Window guard reopened stores"))
    result.ensure_current(pair)
    assert result.scene("17MNR","scene_0")==pair


@pytest.mark.parametrize("change",["artifact","index","revision","receipt"])
def test_bad_pairing_rejected(pair_case,change):
    case=pair_case
    if change=="artifact":
        path=case["quality"]/"provenance/pairings/pair-v2/evidence.json"
        path.write_bytes(path.read_bytes()+b" ")
    else:
        if change=="index": case["links"][0]["spectral_10m_time_index"]=0
        elif change=="revision": case["document"]["spectral_data_revision"]="observed-stale"
        else: case["evidence"]["17MNR_scene_0"]["legacy_receipt_sha256"]="0"*64
        case["publish"]()
    with pytest.raises(CatalogConsistencyError):
        reader(case,pairing="current")


def test_relocated_roots_use_explicit_mapping_without_rewriting_bindings(pair_case,tmp_path):
    case=pair_case
    moved={name:tmp_path/("moved-"+name) for name in ("spectral","quality")}
    for name,target in moved.items(): shutil.copytree(case[name],target)
    before={name:file_hashes(path) for name,path in moved.items()}
    entries=load_catalog(moved["spectral"],"sentinel-2")
    with pytest.raises(CatalogConsistencyError,match="root_map"):
        QualityCompanion(moved["spectral"],moved["quality"],entries,pairing="current")
    mapping={str(case[name]):str(moved[name]) for name in moved}
    result=QualityCompanion(moved["spectral"],moved["quality"],entries,pairing="current",root_map=mapping)
    pair=result.scene("17MNR","scene_0")
    assert pair.quality_20m.zarr_path.is_relative_to(moved["quality"])
    assert before=={name:file_hashes(path) for name,path in moved.items()}


@pytest.mark.parametrize("mode",["legacy","current"])
def test_load_uses_recorded_pixel_evidence_without_reading_image_arrays(pair_case,monkeypatch,mode):
    from zarr.core.array import Array
    original=Array.__getitem__
    def guarded(array,key):
        if array.ndim==3: pytest.fail("Reader load decoded image pixels")
        return original(array,key)
    monkeypatch.setattr(Array,"__getitem__",guarded)
    assert reader(pair_case,pairing=mode).scene("17MNR","scene_0").quality_available


def test_legacy_cannot_bypass_corrupt_present_provenance(pair_case):
    write_json(pair_case["quality"]/"provenance/current.json",{"schema_version":99})
    with pytest.raises(CatalogConsistencyError):
        reader(pair_case,pairing="legacy")


@pytest.mark.parametrize("mode",["legacy","current"])
def test_legacy_artifacts_are_authenticated_before_parquet_decode(pair_case,mode):
    path=pair_case["quality"]/"quality_links.parquet"
    path.write_bytes(path.read_bytes()+b"unpublished change")
    with pytest.raises(CatalogConsistencyError,match="artifact differs"):
        reader(pair_case,pairing=mode)
