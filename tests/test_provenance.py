"""Standalone schema-v1 fixtures and read-only provenance tests."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from grits_resolver import provenance as p
from grits_resolver.catalog import CatalogConsistencyError
from grits_resolver.provenance_contract import (
    canonical_json,
    data_revision,
    write_tables,
)


def publish(root, outputs, *, sources=None, source_links=None, witnesses=None, evidence=None,
            production_scope=None, production_run_id=None, dataset_id="fixture", variables=None, limitations=None, release_version=1):
    """Build immutable metadata fixtures; no producer-repository imports."""
    sources, source_links, witnesses, evidence = sources or [], source_links or [], witnesses or {}, evidence or {}
    payload = {"outputs": outputs, "sources": sources, "source_links": source_links,
               "witnesses": witnesses, "evidence": evidence, "production_scope": production_scope,
               "production_run_id": production_run_id, "dataset_id": dataset_id,
               "variables": variables, "limitations": limitations, "release_version": release_version}
    revision = data_revision(outputs, witnesses)
    release = "meta-" + p.hash_bytes(canonical_json(payload))
    directory = p.safe_relative(root, "provenance/releases/" + release)
    directory.mkdir(parents=True)
    write_tables(directory, outputs, sources, source_links)
    dataset = {"schema_version": release_version, "metadata_release_id": release, "data_revision": revision,
               "delivery_kind": "metadata_only", "dataset_id": dataset_id, "evidence_status": "partial_evidence",
               "execution_kind": "production" if production_scope else "historical_import",
               "counts": {"outputs": len(outputs), "sources": len(sources), "source_links": len(source_links)}}
    if production_scope:
        dataset["production_scope"] = production_scope
    if production_run_id:
        dataset["production_run_id"] = production_run_id
    contents = {"dataset.json": dataset, "variables.json": variables if variables is not None else [{"stored_physical_quantity": None}],
                "limitations.json": limitations if limitations is not None else [{"reason": "fixture unknown upstream"}], "evidence/data_witnesses.json": witnesses,
                **{"evidence/" + name: value for name, value in evidence.items()}}
    for name, value in contents.items():
        path = p.safe_relative(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value))
    manifest = "".join(p.file_digest(path) + "  " + name + "\n" for name, path in sorted(p._files(directory).items()))
    (directory / "FILES.sha256").write_text(manifest, encoding="utf-8")
    pointer = {"schema_version": release_version, "metadata_release_id": release, "data_revision": revision,
               "dataset_sha256": p.file_digest(directory / "dataset.json"),
               "manifest_sha256": p.file_digest(directory / "FILES.sha256")}
    p.safe_relative(root, "provenance/current.json").write_bytes(canonical_json(pointer))
    return pointer


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    store = root / "17MNR/monthly.zarr"
    group = zarr.open_group(str(store), mode="w", zarr_format=3)
    group.attrs.update(grid_id="grid", product_type="smonthly")
    times = np.array(["2025-01-16T01:02:03.123456789", "2025-02-13T04:05:06"], dtype="datetime64[ns]")
    group.create_array("time", data=times, dimension_names=["time"])
    group.create_array("x", data=np.array([15., 45.]), dimension_names=["x"])
    group.create_array("y", data=np.array([45., 15.]), dimension_names=["y"])
    group.create_array("VV_dB", data=np.arange(8,dtype="float32").reshape(2,2,2), dimension_names=["time","y","x"])
    group.create_array("VH_dB", data=np.ones((2,2,2), dtype="float32"), dimension_names=["time","y","x"])
    group.create_array("n_obs", data=np.ones((2,2,2), dtype="uint8"), dimension_names=["time","y","x"])
    rows = [{"output_id": "output-" + str(i), "generation_id": None, "store_path": "17MNR/monthly.zarr",
             "zarr_group": "", "variable_scope": ["VV_dB", "VH_dB", "n_obs"], "tile_id": "17MNR", "grid_id": "grid",
             "product_type": "smonthly", "record_kind": "monthly", "time_index": i,
             "time_value_utc": p._stamp(stamp), "evidence_status": "artifact_observed",
             "unknown_reason": "Historical execution unknown"} for i, stamp in enumerate(times)]
    witnesses = {"17MNR/monthly.zarr/" + name: p.file_digest(path) for name,path in p._files(store).items() if name.endswith("zarr.json")}
    witnesses["17MNR/monthly.zarr/time#values"] = p.hash_bytes(canonical_json(p._times(group["time"])))
    witnesses["17MNR/monthly.zarr/x#values"] = p.hash_bytes(canonical_json(group["x"][:].tolist()))
    return root, store, rows, witnesses


def test_vendored_contract_has_exact_pinned_body():
    path = Path(p.__file__).with_name("provenance_contract.py")
    header, expected, _, body = path.read_bytes().split(b"\n",3)
    assert b"canonical:" in header
    assert expected.split()[-1].decode() == hashlib.sha256(body).hexdigest()


def test_missing_metadata_is_explicit_but_pending_still_blocks(tmp_path):
    root=tmp_path/'missing';root.mkdir()
    view=p.ProvenanceView(root)
    assert view.explain()["metadata_status"] == "missing"
    with pytest.raises(CatalogConsistencyError,match="Required"):
        p.ProvenanceView(root,required=True)
    pending=root/'provenance/pending/x.json';pending.parent.mkdir(parents=True);pending.write_text('{}')
    with pytest.raises(CatalogConsistencyError,match="pending"):
        view.assert_current()
    with pytest.raises(CatalogConsistencyError,match="pending"):
        p.ProvenanceView(root)


def test_historical_times_metadata_and_read_only_moved_roots(dataset,tmp_path):
    root,store,rows,witnesses=dataset
    meta=tmp_path/'metadata'
    publish(meta,rows,witnesses=witnesses)
    before={n:p.file_digest(f) for n,f in p._files(root).items()}
    view=p.ProvenanceView(root,metadata_root=meta,required=True)
    result=view.explain(store_path=store,zarr_group="",time_index=0)
    assert result['matched'] and result['outputs'][0]['output']['generation_id'] is None
    assert result['outputs'][0]['output']['time_value_utc'].endswith('03.123456789Z')
    assert before=={n:p.file_digest(f) for n,f in p._files(root).items()}
    moved=tmp_path/'moved';shutil.copytree(root,moved/'data');shutil.copytree(meta,moved/'meta')
    assert p.ProvenanceView(moved/'data',metadata_root=moved/'meta').explain(time_index=1)['matched']


def test_exact_generation_sources_and_explanation(dataset):
    root,_,rows,witnesses=dataset
    rows[0].update(generation_id='g1',run_id='r1',processing_id='p1',evidence_status='execution_recorded',evidence_ref='evidence/run.json')
    source={'source_id':'s1','product_id':'asset','href':'https://example.invalid/raw.tif','evidence_status':'source_declared','evidence_ref':'evidence/run.json'}
    link={'output_id':rows[0]['output_id'],'generation_id':'g1','source_id':'s1','role':'used','evidence_ref':'evidence/run.json'}
    publish(root,rows,sources=[source],source_links=[link],witnesses=witnesses,evidence={'run.json':{'scale_applied':False}})
    view=p.ProvenanceView(root)
    result=view.explain(output_id=rows[0]['output_id'])
    assert result['outputs'][0]['sources'][0]['link']['generation_id']=='g1'
    assert result['evidence']['run.json']['scale_applied'] is False
    assert view.explain(output_id='absent')['matched'] is False


@pytest.mark.parametrize('kind',['manifest','table','extra','pointer'])
def test_release_tampering_rejected(dataset,kind):
    root,_,rows,witnesses=dataset
    pointer=publish(root,rows,witnesses=witnesses)
    release=p.safe_relative(root,'provenance/releases/'+pointer['metadata_release_id'])
    if kind=='pointer':
        current=root/'provenance/current.json';x=json.loads(current.read_text());x['schema_version']=2;current.write_text(json.dumps(x))
    else:
        target=release/({'manifest':'FILES.sha256','table':'outputs.parquet','extra':'extra.json'}[kind])
        with target.open('ab') as f:f.write(b'changed')
    with pytest.raises(CatalogConsistencyError):p.ProvenanceView(root)


@pytest.mark.parametrize('kind',['time','grid','catalog','pointer'])
def test_changed_source_or_version_rejects_until_reload(dataset,kind):
    root,store,rows,witnesses=dataset
    (root/'catalog.parquet').write_bytes(b'initial')
    publish(root,rows,witnesses=witnesses)
    view=p.ProvenanceView(root)
    if kind=='time':
        zarr.open_group(str(store),mode='a')['time'][0]=np.datetime64('2025-01-01','ns')
    elif kind=='grid':
        zarr.open_group(str(store),mode='a').attrs['grid_id']='other'
    elif kind=='catalog':
        (root/'catalog.parquet').write_bytes(b'changed')
    else:
        rows[0].update(generation_id='g2',run_id='r2',processing_id='p2')
        publish(root,rows,witnesses=witnesses)
    with pytest.raises(CatalogConsistencyError):view.assert_current(stores=[store])
    if kind in {'time','grid'}:
        with pytest.raises(CatalogConsistencyError):p.ProvenanceView(root)


def test_window_guard_uses_no_hash_or_zarr_reopening(dataset,monkeypatch):
    root,store,rows,witnesses=dataset
    publish(root,rows,witnesses=witnesses)
    view=p.ProvenanceView(root)
    monkeypatch.setattr(p,'file_digest',lambda *a:pytest.fail('per-window hashing'))
    monkeypatch.setattr(zarr,'open_group',lambda *a,**kw:pytest.fail('per-window Zarr reopen'))
    for _ in range(3):view.assert_current(stores=[store])
    assert view.explain(store_path=store,time_index=0)['matched']


@pytest.mark.parametrize('bad',['../x','/tmp/x','a\\b','a/./b','a//b','C:/x'])
def test_relative_path_escape_rejected(tmp_path,bad):
    with pytest.raises(CatalogConsistencyError):p.safe_relative(tmp_path,bad)


def test_coordinate_load_is_cached_per_group(dataset,monkeypatch):
    root,_,rows,witnesses=dataset
    publish(root,rows,witnesses=witnesses)
    original=p._times;calls=[]
    def counted(array):
        calls.append(array.path)
        return original(array)
    monkeypatch.setattr(p,'_times',counted)
    p.ProvenanceView(root)
    assert calls==['time','time']  # Once for coordinate witness, once for row mapping; never once per output.


def test_catalog_guard_detects_change_during_snapshot_load(dataset,monkeypatch):
    root,_,rows,witnesses=dataset
    catalog=root/'catalog.parquet';catalog.write_bytes(b'before')
    publish(root,rows,witnesses=witnesses)
    original=p.read_snapshot
    def changed(*args):
        result=original(*args)
        catalog.write_bytes(b'after')
        return result
    monkeypatch.setattr(p,'read_snapshot',changed)
    with pytest.raises(CatalogConsistencyError,match='changed'):
        p.ProvenanceView(root)


def test_s1_managed_catalog_plan_is_checked_at_load(dataset):
    root,_,rows,witnesses=dataset
    catalog=root/'catalog.parquet';catalog.write_bytes(b'published')
    plan={'targets':{'catalog.parquet':{'desired_sha256':p.file_digest(catalog)}}}
    publish(root,rows,witnesses=witnesses,production_scope='s1_monthly_workflow',production_run_id='r1',
            evidence={'runs/r1/catalog_plan.json':plan})
    p.ProvenanceView(root)
    catalog.write_bytes(b'changed')
    with pytest.raises(CatalogConsistencyError,match='Catalog differs'):
        p.ProvenanceView(root)


def test_scl_uses_authenticated_acceptance_without_scanning_image_chunks(tmp_path,monkeypatch):
    root=tmp_path/'quality';store=root/'17MNR/native.zarr'
    top=zarr.open_group(str(store),mode='w',zarr_format=3)
    group=top.create_group('20m');group.attrs['grid_id']='g20'
    times=np.array(['2025-01-15'],dtype='datetime64[ns]')
    group.create_array('time',data=times,dimension_names=['time'])
    values=np.array([[[0,9],[4,11]]],dtype='uint8')
    group.create_array('scl',data=values,dimension_names=['time','y','x'])
    row={'output_id':'quality-o','generation_id':'q1','run_id':'r1','processing_id':'copy',
         'store_path':'17MNR/native.zarr','zarr_group':'20m','variable_scope':['scl'],
         'grid_id':'g20','tile_id':'17MNR','product_type':'quality_scl','record_kind':'scene',
         'time_index':0,'time_value_utc':'2025-01-15T00:00:00Z','source_item_id':'scene1',
         'evidence_status':'execution_recorded','evidence_ref':'evidence/quality/receipt.json'}
    h=hashlib.sha256(json.dumps({'dtype':'uint8','shape':[2,2],'block':512},sort_keys=True).encode('ascii'))
    h.update(values[0].tobytes())
    receipt={'status':'ACCEPTED','generation_id':'q1','item_id':'scene1',
             'verification':{'full_pixel_comparison':True,'mismatched_pixels':0,
                             'decoded_hash_scheme':'uint8-block512-v1','decoded_sha256':h.hexdigest()},
             'link':{'quality_20m_store':row['store_path'],'quality_20m_time_index':0,'quality_20m_grid_id':'g20'}}
    publish(root,[row],witnesses={'decoded/17MNR/native.zarr/20m/scl/scene1':h.hexdigest()},
            evidence={'quality/receipt.json':receipt},production_scope='native_scl_accepted_copies')
    original=zarr.Array.__getitem__
    def no_image_reads(array,key):
        if array.ndim==3:pytest.fail('P4 repeated P3 full image verification')
        return original(array,key)
    monkeypatch.setattr(zarr.Array,'__getitem__',no_image_reads)
    view=p.ProvenanceView(root)
    assert 'no fresh image chunk' in view.explain(time_index=0)['verification_scope']
    # A metadata claim contradicting its accepted receipt is refused.
    receipt['verification']['decoded_sha256']='0'*64
    publish(root,[row],witnesses={'decoded/17MNR/native.zarr/20m/scl/scene1':h.hexdigest()},
            evidence={'quality/receipt.json':receipt},production_scope='native_scl_accepted_copies')
    with pytest.raises(CatalogConsistencyError,match='witness'):
        p.ProvenanceView(root)


def _numeric_receipt(row, generation="g1"):
    row.update(generation_id=generation, run_id="run-" + generation,
               processing_id="processing", evidence_status="execution_recorded",
               evidence_ref="evidence/runs/" + generation + "/receipt.json")
    steps, outputs = [], {}
    for variable in ("VV_dB", "VH_dB", "n_obs"):
        count = variable == "n_obs"
        domain, dtype = ("count", "uint8") if count else ("db", "float32")
        operations = ([('observation_count', 'linear_power', 'count',
                        {'method': 'sum_isfinite', 'count_unit': 'aligned_scene_stack_entry'})] if count else [
            ('monthly_composite', 'linear_power', 'linear_power',
             {'method': 'nanmedian', 'backend': 'numpy', 'axis': 0, 'nan_policy': 'ignore_nan'}),
            ('linear_to_db', 'linear_power', 'db', {'formula': '10*log10(x)', 'nonpositive': 'NaN'})])
        operations.append(('storage_cast_and_write', domain, domain, {'dtype': dtype}))
        steps.extend({"operation": op, "variable": variable, "input_domain": before, "output_domain": after,
                      "parameters": parameters, "status": "executed", "calls": 1}
                     for op, before, after, parameters in operations)
        outputs[variable] = {"domain": domain, "dtype": dtype, "unit": "aligned_scene_observations" if count else "dB",
                             "physical_quantity": "finite_VV_stack_entries_from_selected_track" if count else None,
                             "nodata": 0 if count else "NaN", "applied_scale": 1.0, "applied_offset": 0.0}
    return {
        "output_id": row["output_id"], "generation_id": generation,
        "numeric_processing": {
            "schema_version": 1, "scope": "monthly_writer_input_to_zarr",
            "input": {"actual_domain": None, "operator_expected_domain": "linear_power",
                      "physical_quantity": None, "evidence_status": "unknown",
                      "unknown_reason": "Source units are not independently established."},
            "steps": steps, "output": outputs,
        },
    }


def test_legacy_explanation_does_not_borrow_other_output_numeric_record(dataset):
    root, _, rows, witnesses = dataset
    receipt = _numeric_receipt(rows[1])
    rows[0]["evidence_ref"] = "evidence/legacy.json"
    publish(root, rows, witnesses=witnesses,
            evidence={"legacy.json": {"record_kind": "historical"},
                      "runs/g1/receipt.json": receipt})
    view = p.ProvenanceView(root)
    old = view.explain(output_id=rows[0]["output_id"])["numeric_processing"]
    assert old == [{"output_id": rows[0]["output_id"], "generation_id": None,
                    "record": None, "unknown_reason": "No numerical processing record is attached to this output generation."}]
    assert view.explain(output_id="absent")["numeric_processing"] == []
    assert view.explain(time_index=1)["numeric_processing"][0]["record"] == receipt["numeric_processing"]


def test_numeric_explanation_reads_exact_receipt_from_detached_companion(dataset, tmp_path, monkeypatch):
    root, _, rows, witnesses = dataset
    receipt = _numeric_receipt(rows[0])
    metadata = tmp_path / "numeric-metadata"
    publish(metadata, rows, witnesses=witnesses, evidence={"runs/g1/receipt.json": receipt})
    before = {n: p.file_digest(f) for n, f in p._files(root).items()}
    original = zarr.Array.__getitem__

    def coordinates_only(array, key):
        if array.ndim >= 2:
            pytest.fail("Numeric explanation must not read image pixels")
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", coordinates_only)
    result = p.ProvenanceView(root, metadata_root=metadata, required=True).explain(time_index=0)
    assert result["numeric_processing"] == [{
        "output_id": rows[0]["output_id"], "generation_id": "g1",
        "record": receipt["numeric_processing"], "unknown_reason": None,
    }]
    assert result["numeric_processing"][0]["record"]["input"]["actual_domain"] is None
    assert before == {n: p.file_digest(f) for n, f in p._files(root).items()}


def test_replaced_numeric_record_is_not_taken_from_prior_generation(dataset):
    root, _, rows, witnesses = dataset
    first_receipt = _numeric_receipt(rows[0], "old")
    other_receipt = _numeric_receipt(rows[1], "other-month")
    evidence = {"runs/old/receipt.json": first_receipt,
                "runs/other-month/receipt.json": other_receipt}
    first = publish(root, rows, witnesses=witnesses, evidence=evidence)
    original = p.ProvenanceView(root)
    old_release = p.safe_relative(root, "provenance/releases/" + first["metadata_release_id"])
    old_files = {n: p.file_digest(f) for n, f in p._files(old_release).items()}
    new_receipt = _numeric_receipt(rows[0], "replacement")
    new_receipt["numeric_processing"]["steps"][0]["calls"] = 3
    evidence["runs/replacement/receipt.json"] = new_receipt
    publish(root, rows, witnesses=witnesses, evidence=evidence)
    with pytest.raises(CatalogConsistencyError, match="version changed"):
        original.explain(time_index=0)
    current = p.ProvenanceView(root).explain()
    by_output = {r["output_id"]: r for r in current["numeric_processing"]}
    assert by_output[rows[0]["output_id"]]["generation_id"] == "replacement"
    assert by_output[rows[0]["output_id"]]["record"] == new_receipt["numeric_processing"]
    assert by_output[rows[1]["output_id"]]["record"] == other_receipt["numeric_processing"]
    assert old_files == {n: p.file_digest(f) for n, f in p._files(old_release).items()}


@pytest.mark.parametrize("invalid", [
    "output_id", "generation_id", "missing_generation", "unbound_output",
    "schema_version", "boolean_version", "missing_version", "scope", "null_record", "list_record",
])
def test_numeric_explanation_rejects_wrong_binding_or_version(dataset, invalid):
    root, _, rows, witnesses = dataset
    receipt = _numeric_receipt(rows[0])
    if invalid in {"output_id", "generation_id"}:
        receipt[invalid] = "other"
    elif invalid == "missing_generation":
        receipt.pop("generation_id")
    elif invalid == "unbound_output":
        rows[0].update(generation_id=None, run_id=None, processing_id=None,
                       evidence_status="artifact_observed")
        receipt["generation_id"] = None
    elif invalid == "schema_version":
        receipt["numeric_processing"]["schema_version"] = 2
    elif invalid == "boolean_version":
        receipt["numeric_processing"]["schema_version"] = True
    elif invalid == "missing_version":
        receipt["numeric_processing"].pop("schema_version")
    elif invalid == "scope":
        receipt["numeric_processing"]["scope"] = "upstream_assumed"
    elif invalid == "null_record":
        receipt["numeric_processing"] = None
    else:
        receipt["numeric_processing"] = []
    publish(root, rows, witnesses=witnesses, evidence={"runs/g1/receipt.json": receipt})
    view = p.ProvenanceView(root)
    with pytest.raises(CatalogConsistencyError, match="numerical processing|Numerical processing"):
        view.explain(output_id=rows[0]["output_id"])
    assert view.explain(output_id=rows[1]["output_id"])["numeric_processing"][0]["record"] is None


@pytest.mark.parametrize("change", ["missing_input", "false_domain", "empty_steps", "wrong_formula", "wrong_dtype"])
def test_numeric_explanation_uses_the_producer_contract(dataset, change):
    root, _, rows, witnesses = dataset
    receipt = _numeric_receipt(rows[0])
    record = receipt["numeric_processing"]
    if change == "missing_input":
        record.pop("input")
    elif change == "false_domain":
        record["input"]["actual_domain"] = "linear_power"
    elif change == "empty_steps":
        record["steps"] = []
    elif change == "wrong_formula":
        next(s for s in record["steps"] if s["operation"] == "linear_to_db")["parameters"]["formula"] = "20*log10(x)"
    else:
        record["output"]["VV_dB"]["dtype"] = "uint16"
    publish(root, rows, witnesses=witnesses, evidence={"runs/g1/receipt.json": receipt})
    with pytest.raises(CatalogConsistencyError, match="numerical processing"):
        p.ProvenanceView(root).explain(time_index=0)


def test_mutating_explanation_does_not_mutate_loaded_evidence(dataset):
    root, _, rows, witnesses = dataset
    receipt = _numeric_receipt(rows[0])
    publish(root, rows, witnesses=witnesses, evidence={"runs/g1/receipt.json": receipt})
    before = {n: p.file_digest(f) for n, f in p._files(root).items()}
    view = p.ProvenanceView(root)
    result = view.explain(time_index=0)
    result["numeric_processing"][0]["record"]["input"]["actual_domain"] = "consumer-edit"
    result["evidence"]["runs/g1/receipt.json"]["generation_id"] = "consumer-edit"
    result["outputs"][0]["output"]["run_id"] = "consumer-edit"
    again = view.explain(time_index=0)
    assert again["numeric_processing"][0]["record"] == receipt["numeric_processing"]
    assert again["outputs"][0]["output"]["run_id"] == rows[0]["run_id"]
    assert before == {n: p.file_digest(f) for n, f in p._files(root).items()}


@pytest.mark.parametrize("generation", [None, "replaced-generation"])
def test_historical_annotation_explanation_is_generation_scoped(dataset, generation):
    root, _, rows, witnesses = dataset
    rows[0]["generation_id"] = generation
    rows[0]["evidence_ref"] = rows[1]["evidence_ref"] = "evidence/store.json"
    scope = {rows[0]["output_id"]: None}
    historical = {"historical_processing_ref": "evidence/confirmation.json"}
    publish(root, rows, witnesses=witnesses,
            variables=[{**historical, "units": "dB", "historical_output_generations": scope,
                        "declared_attributes": {"units": "provider-declared"}}],
            limitations=[{**historical, "historical_output_generations": scope}],
            evidence={"store.json": {"historical_context": {**historical, "output_generations": scope}},
                      "confirmation.json": {"basis": "producer_confirmation"}})
    view = p.ProvenanceView(root)
    result = view.explain(time_index=0)
    retained = generation is None
    assert ("units" in result["variables"][0]) == retained
    assert ("historical_processing_ref" in result["limitations"][0]) == retained
    assert ("historical_context" in result["evidence"]["store.json"]) == retained
    assert result["variables"][0]["declared_attributes"]["units"] == "provider-declared"
    result["variables"][0]["historical_output_generations"] = {"consumer-edit": None}
    assert view.explain(time_index=0)["variables"][0].get("historical_output_generations") == (scope if retained else None)
    for selection in ({"time_index": 1}, {"output_id": "absent"}):
        other = view.explain(**selection)
        assert "historical_processing_ref" not in other["variables"][0]
        assert "historical_processing_ref" not in other["limitations"][0]
        assert all("historical_context" not in item for item in other["evidence"].values())


@pytest.mark.parametrize("scope", [["output-0"], {"output-0": 42}])
@pytest.mark.parametrize("location", ["variables", "limitations", "evidence"])
def test_historical_annotation_rejects_malformed_scope(dataset, scope, location):
    root, _, rows, witnesses = dataset
    rows[0]["evidence_ref"] = "evidence/store.json"
    if location == "evidence":
        annotations = {"evidence": {"store.json": {"historical_context": {"output_generations": scope}}}}
    else:
        annotations = {location: [{"historical_output_generations": scope}], "evidence": {"store.json": {}}}
    publish(root, rows, witnesses=witnesses, **annotations)
    with pytest.raises(CatalogConsistencyError, match="historical annotation"):
        p.ProvenanceView(root).explain(time_index=0)


@pytest.mark.parametrize("invalid", [None, "scope_missing", "empty_scope", "unknown_output", "wrong_generation",
                                      "wrong_store", "wrong_group", "wrong_variable", "missing_evidence", "bool_version"])
def test_v2_validates_scope_at_load_without_changing_table_revision(dataset, invalid):
    root, _, rows, witnesses = dataset
    row = rows[0]
    variable = {"store_path": row["store_path"], "zarr_group": "", "variable": "VV_dB",
                "units": "dB", "historical_processing_ref": "evidence/confirmation.json",
                "historical_output_generations": {row["output_id"]: row["generation_id"]}}
    if invalid == "scope_missing":
        variable.pop("historical_output_generations")
    elif invalid == "empty_scope":
        variable["historical_output_generations"] = {}
    elif invalid == "unknown_output":
        variable["historical_output_generations"] = {"not-current": None}
    elif invalid == "wrong_generation":
        variable["historical_output_generations"][row["output_id"]] = "old-generation"
    elif invalid == "wrong_store":
        variable["store_path"] = "other/store.zarr"
    elif invalid == "wrong_group":
        variable["zarr_group"] = "other-group"
    elif invalid == "wrong_variable":
        variable["variable"] = "absent_band"
    elif invalid == "missing_evidence":
        variable["historical_processing_ref"] = "evidence/absent.json"
    pointer = publish(root, rows, witnesses=witnesses, variables=[variable],
                      evidence={"confirmation.json": {"basis": "synthetic historical fixture"}},
                      release_version=True if invalid == "bool_version" else 2)
    if invalid:
        with pytest.raises(CatalogConsistencyError):
            p.ProvenanceView(root)
    else:
        view = p.ProvenanceView(root)
        assert pointer["data_revision"] == data_revision(rows, witnesses)
        assert view.explain(time_index=0)["variables"][0]["units"] == "dB"
        assert "units" not in view.explain(time_index=1)["variables"][0]
