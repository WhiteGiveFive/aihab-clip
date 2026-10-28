"""Bypass declaration, source authentication, reuse and immutable selection."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from test_attention_fusion_comparison import study as parent_study, reseal, write_json
from multimodal import attention_fusion_comparison as old
from multimodal import attention_fusion_bypass_comparison as bypass
from multimodal.attention_fusion_bypass_protocol import build_matrix, expected_config
from multimodal.attention_fusion_bypass_training import _source_hashes
from multimodal.attention_fusion_training import configuration_hash, sha256_file


@pytest.fixture(scope="module")
def extension(parent_study):
    repo, parent, parent_matrix, old_references, old_declaration = parent_study
    import tools.run_multimodal_attention_fusion as runner
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "read_frozen_plan", lambda *a, **k: (parent, parent_matrix, {"manifest_payload_sha256":"synthetic"}))
        patch.setattr(old, "load_reference_inputs", lambda *a, **k: old_references)
        old.analyze_study(parent, parent_matrix, old_declaration,
                         repo / parent["paths"]["artifact_root"] / "reports/m3_comparison_v1", repo_root=repo)
    config = expected_config(parent)
    for c in config["candidates"]:
        c["expected_trainable_parameters"] = 2
    matrix = build_matrix(config)
    for fit in matrix["development_fits"]:
        fit.update(epochs=2, checkpoint_epoch=2)
    declaration = repo / config["paths"]["artifact_root"] / "protocol/contrasts.json"
    bypass.freeze_contrasts(config, matrix, declaration, repo_root=repo)
    references = {key: copy.deepcopy(value) for key,value in old_references.items() if key[1]=="image"}
    for ref in references.values():
        ref["provenance"].update(protocol_config_sha256=configuration_hash(config),
                                 source_protocol_config_sha256=configuration_hash(parent))
    for fit in matrix["development_fits"]:
        source = next(f for f in parent_matrix["development_fits"] if f["candidate_id"]=="attention_cls"
                      and f["seed"]==fit["seed"] and f["learning_rate"]==fit["learning_rate"]
                      and f["geo_preprocessing"]==fit["geo_preprocessing"])
        target = repo / fit["output_dir"]
        shutil.copytree(repo/source["output_dir"],target)
        spec=json.loads((target/"run_spec.json").read_text())
        spec.update(config_sha256=configuration_hash(config),study_id=config["study_id"],candidate_id=fit["candidate_id"],
                    source_hashes=_source_hashes(),input_provenance=references[(fit["seed"],"image")]["provenance"])
        write_json(target/"config.json",config)
        write_json(target/"run_spec.json",spec)
        identity=configuration_hash(spec)
        for name in ("run.json","manifest.json"):
            value=json.loads((target/name).read_text())
            value["run_identity_sha256"]=identity
            write_json(target/name,value)
        reseal(target)
    return repo,config,matrix,references,declaration,parent,parent_matrix,old_references


def test_exact_37_contrasts_two_native_primaries_and_all_directions(extension):
    _,config,matrix,*_=extension
    d=bypass.declare_contrasts(config,matrix)
    assert len(d["contrasts"])==37
    primary=[r for r in d["contrasts"] if r["kind"]=="primary"]
    assert [r["contrast_id"] for r in primary]==d["primary_contrast_ids"]
    assert {r["lhs_arm_id"] for r in primary}=={f"{name}__geo_native" for name in bypass.BYPASSES}
    assert {r["rhs_arm_id"] for r in primary}=={"attention_cls__geo_native"}
    assert len([r for r in d["contrasts"] if r["kind"]=="preprocessing"])==4
    assert len([r for r in d["contrasts"] if r["kind"]=="baseline_image"])==6
    direct=[r for r in d["contrasts"] if r["kind"]=="bypass_direct"]
    assert len(direct)==3 and all(r["lhs_arm_id"].startswith("attention_cls_skip_original") for r in direct)
    assert "after original M3" in d["claim_scope"]
    assert "no isolated" in d["attribution"]


def test_declaration_cannot_change_or_be_created_after_any_new_fit(extension):
    repo,config,matrix,_,declaration,*_=extension
    before=declaration.read_bytes()
    bypass.freeze_contrasts(config,matrix,declaration,repo_root=repo)
    assert declaration.read_bytes()==before
    with pytest.raises(old.ComparisonError,match="before extension development fits"):
        bypass.freeze_contrasts(config,matrix,repo/"late.json",repo_root=repo)
    changed=copy.deepcopy(config)
    changed["interpretation"]["minimum_practical_effect"]=0.05
    with pytest.raises(old.ComparisonError,match="declaration differs"):
        bypass.freeze_contrasts(changed,matrix,declaration,repo_root=repo)


def test_new72_bundles_are_authenticated_with_separate_sources(extension):
    repo,config,matrix,references,*_=extension
    records=bypass.authenticate_runs(config,matrix,references,repo_root=repo)
    rates,selected=old.select_learning_rates(config,matrix,records)
    assert (len(records),len(rates),len(selected))==(72,18,6)
    assert all(set(a["per_seed"])=={"1","2","3","4"} for a in selected)
    with pytest.raises(old.ComparisonError,match="Run recipe differs"):
        old.authenticate_runs(config,matrix,references,repo_root=repo)


@pytest.mark.parametrize("violation",["sources","validation_identity","scaler","saved_metric"])
def test_bypass_authentication_rejects_resealed_wrong_evidence(extension,violation):
    repo,config,matrix,references,*_=extension
    directory=repo/matrix["development_fits"][0]["output_dir"]
    names={"sources":"run_spec.json","validation_identity":"run_spec.json",
           "scaler":"geo_preprocessing.json","saved_metric":"metrics.json"}
    name=names[violation]
    saved={n:(directory/n).read_bytes() for n in (name,"run.json","manifest.json")}
    try:
        value=json.loads((directory/name).read_text())
        if violation=="sources":
            value["source_hashes"]=old._source_hashes()
        elif violation=="validation_identity":
            value["input_hashes"]["validation"]["row_uid"]="wrong"
        elif violation=="scaler":
            value["fit_row_sha256"]="wrong"
        else:
            value["validation"]["acc1"]+=0.1
        write_json(directory/name,value)
        if name=="run_spec.json":
            for filename in ("run.json","manifest.json"):
                row=json.loads((directory/filename).read_text())
                row["run_identity_sha256"]=configuration_hash(value)
                write_json(directory/filename,row)
        reseal(directory)
        with pytest.raises(old.ComparisonError):
            bypass.authenticate_runs(config,matrix,references,repo_root=repo)
    finally:
        for name,data in saved.items():
            (directory/name).write_bytes(data)


def test_cross_study_pairing_rejects_identity_or_environment_drift():
    base={"environment":{"device":"cuda:0"},"predictions":{
        "row_uid":np.array(["a"]),"label":np.array([0]),"plot_idx":np.array(["p"])}}
    bypass._assert_shared_evidence([base],[copy.deepcopy(base)])
    changed=copy.deepcopy(base)
    changed["predictions"]["plot_idx"][0]="q"
    with pytest.raises(old.ComparisonError,match="identities"):
        bypass._assert_shared_evidence([base],[changed])
    changed=copy.deepcopy(base)
    changed["environment"]["device"]="cpu"
    with pytest.raises(old.ComparisonError,match="environments"):
        bypass._assert_shared_evidence([base],[changed])


def test_complete_analysis_retains25_arms37_contrasts_but_publishes_only24_new_refs(extension,monkeypatch):
    repo,config,matrix,references,declaration,parent,parent_matrix,old_references=extension
    import multimodal.attention_fusion_bypass_protocol as protocol
    import tools.run_multimodal_attention_fusion as runner
    oldroot=repo/parent["paths"]["artifact_root"]
    before={str(p):sha256_file(p) for p in oldroot.rglob('*') if p.is_file()}
    monkeypatch.setattr(protocol,"read_frozen_plan",lambda *a,**k:(config,matrix,{
        "manifest_payload_sha256":"synthetic","files":{"contrasts.json":sha256_file(declaration)}}))
    monkeypatch.setattr(protocol,"verify_parent_bindings",lambda *a,**k:{"synthetic":True})
    def parent_plan(path,*a,**k):
        assert isinstance(path,Path)
        return parent,parent_matrix,{}
    monkeypatch.setattr(runner,"read_frozen_plan",parent_plan)
    monkeypatch.setattr(old,"load_reference_inputs",lambda *a,**k:old_references)
    monkeypatch.setattr(bypass,"load_reference_inputs",lambda *a,**k:references)
    output=repo/config["paths"]["artifact_root"]/"reports/comparison"
    result=bypass.analyze_study(config,matrix,declaration,output,repo_root=repo)
    assert result["counts"]=={"development_fits":72,"reused_development_fits":228,
        "total_authenticated_fits":300,"new_selected_arms":6,"reused_selected_arms":19,
        "selected_arms":25,"selected_checkpoints":24,"reused_selected_checkpoints":76,"contrasts":37}
    assert len(result["primary_contrast_ids"])==2
    for contrast in result["contrasts"]:
        if contrast["kind"] in {"primary","bypass_direct"}:
            assert contrast["mean_difference"]==dict.fromkeys(old.METRICS,0.0)
            assert contrast["category"]=="no_improvement"
    selection=json.loads((output/"selection.json").read_text())
    assert len(selection["selected_references"])==24 and len(selection["reused_selected_references"])==76
    assert selection["config_sha256"]==configuration_hash(config)
    for ref in selection["selected_references"]:
        assert json.loads((repo/ref["selection_path"]).read_text())==ref
        assert config["paths"]["artifact_root"] in ref["bundle_path"]
    assert all("confusion_matrix" not in row["validation"] for a in result["selected_arms"]
               for history in a["histories_by_seed"].values() for row in history)
    assert all("confusion_matrix" in seed["metrics"] for a in result["selected_arms"] for seed in a["per_seed"].values())
    assert before=={str(p):sha256_file(p) for p in oldroot.rglob('*') if p.is_file()}
    with np.load(output/"bootstrap_replicates.npz") as saved:
        assert saved["differences"].shape==(31,37,4)
        assert saved["plot_multiplicities"].shape==(31,3)
    with pytest.raises(old.ComparisonError,match="Analysis exists"):
        bypass.analyze_study(config,matrix,declaration,output,repo_root=repo)
