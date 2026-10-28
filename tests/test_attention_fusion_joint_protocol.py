"""The M3J declaration freezes the experiment budget, contrasts and provenance."""

from collections import Counter
from copy import deepcopy
import json

import pytest
import yaml

from multimodal import attention_fusion_joint_protocol as protocol
from multimodal.attention_fusion_training import configuration_hash, sha256_file


@pytest.fixture
def config():
    return protocol.load_config()


def test_default_recipe_matches_approved_epoch_batch_and_precision_contract(config):
    assert config["study_id"] == protocol.DIRECT_STUDY_ID
    assert "numerical_policy" not in config
    assert config["verification_policy"] == protocol.VERIFICATION_POLICY
    assert config["result_assurance"] == protocol.PROVISIONAL_ASSURANCE
    assert config["checkpoint"]["automatic_resume"] is False
    assert config["checkpoint"]["prediction_replay"] == "load_and_evaluate"
    training = config["training"]
    assert training["epochs"] == training["scheduler_t_max"] == 50
    assert training["batch_size"] == 16 and training["gradient_accumulation_steps"] == 1
    assert training["optimizer"] == "adam" and training["learning_rate"] == 5e-5
    assert training["weight_decay"] == 0 and training["scheduler_eta_min"] == 0
    assert training["scheduler_step"] == "end_of_epoch"
    assert training["parameter_dtype"] == "float32" and training["autocast_dtype"] == "float16"
    assert not training["grad_scaler"] and not training["validation_autocast"]
    assert training["cudnn_allow_tf32"] and not training["matmul_allow_tf32"]
    assert not training["early_stopping"] and not training["warmup_epochs"]
    assert config["checkpoint"]["best_order"] == ["acc1", "weighted_f1", "mcc", "earliest_epoch"]
    assert config["geo"]["preprocessing"] == "standardize" and config["geo"]["fit_role"] == "train"
    assert config["visual"]["trainable_blocks"] == list(range(17, 27))
    assert config["refit_policy"]["epochs"] == 50
    assert not config["refit_policy"]["select_best_during_all_development_refit"]
    assert not config["refit_policy"]["test_guides_selection"]


def test_geo_only_authorizes_one_shared_fit_with_unchanged_recipe(config):
    geo = protocol.load_config(protocol.GEO_ONLY_CONFIG)
    assert geo["training"] == config["training"]
    assert geo["geo"] == config["geo"]
    assert protocol.package_name(geo) == "seed1_geo_only"
    assert len(protocol.build_matrix(geo)) == 132
    assert protocol.authorized_fits(geo) == [dict(
        run_id="seed1/geo_only", seed=1, regime="geo_only",
        candidate_id="geo_only", initialization="not_applicable")]


@pytest.mark.parametrize("change", ["initializations", "regime", "references"])
def test_geo_only_scope_cannot_expand_or_lose_references(tmp_path, change):
    config = protocol.load_config(protocol.GEO_ONLY_CONFIG)
    if change == "initializations":
        config["authorized_scope"]["initializations"] = ["pretrained", "cs_adapted"]
    elif change == "regime":
        config["authorized_scope"]["regime"] = "joint_partial"
    else:
        config["reporting"]["completed_reference_roots"].pop()
    path = tmp_path / "bad_geo.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        protocol.load_config(path)


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_later_geo_deferral_retains_recipe_full_design_and_one_fit(config, seed):
    geo = protocol.load_config(protocol.package_config_path(seed, "geo_only"))
    assert protocol.geo_reference_regimes(geo) == ("joint_partial",)
    assert geo["reporting"]["deferred_reference_regimes"] == ["frozen_online"]
    assert len(protocol.build_matrix(geo)) == 132
    assert len(protocol.authorized_fits(geo)) == 1
    assert geo["checkpoint"] == config["checkpoint"]
    assert geo["geo"] == config["geo"]


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_later_geo_can_still_require_both_packages_when_not_deferred(tmp_path, seed):
    config = protocol.load_config(protocol.package_config_path(seed, "geo_only"))
    config["reporting"].pop("deferred_reference_regimes")
    config["reporting"]["completed_reference_roots"].append(
        f"multimodal_artifacts/attention_fusion/cs/gse_100m/{protocol.package_identity(seed, 'frozen_online')}")
    path = tmp_path / "full_geo.yaml"
    path.write_text(yaml.safe_dump(config))
    assert protocol.geo_reference_regimes(protocol.load_config(path)) == ("joint_partial", "frozen_online")


@pytest.mark.parametrize("deferred", [[], ["joint_partial"], ["geo_only"],
    ["joint_partial", "frozen_online"], ["frozen_online", "frozen_online"], "frozen_online", None])
def test_geo_deferral_must_be_the_explicit_approved_regime(tmp_path, deferred):
    config = protocol.load_config(protocol.package_config_path(2, "geo_only"))
    config["reporting"]["deferred_reference_regimes"] = deferred
    path = tmp_path / "invalid_geo.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="explicitly defer frozen-online"):
        protocol.load_config(path)


@pytest.mark.parametrize("seed,regime", [(1, "geo_only"), (2, "joint_partial"), (2, "frozen_online")])
def test_deferral_cannot_change_seed1_or_image_package_scope(tmp_path, seed, regime):
    config = protocol.load_config(protocol.package_config_path(seed, regime))
    config["reporting"]["deferred_reference_regimes"] = ["frozen_online"]
    path = tmp_path / "invalid_scope.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="seed 2–4"):
        protocol.load_config(path)


@pytest.mark.parametrize("change", ["missing", "wrong_seed", "extra_deferred", "implicit_deferral"])
def test_geo_deferral_cannot_skip_or_change_active_reference(tmp_path, change):
    config = protocol.load_config(protocol.package_config_path(2, "geo_only"))
    reporting = config["reporting"]
    if change == "missing":
        reporting["completed_reference_roots"] = []
    elif change == "wrong_seed":
        reporting["completed_reference_roots"][0] = reporting["completed_reference_roots"][0].replace("seed2", "seed3")
    elif change == "extra_deferred":
        reporting["completed_reference_roots"].append(
            "multimodal_artifacts/attention_fusion/cs/gse_100m/attention_fusion_joint_frozen_online_seed2_direct_v1")
    else:
        reporting.pop("deferred_reference_regimes")
    path = tmp_path / "invalid_references.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="every active completed image package"):
        protocol.load_config(path)
    with pytest.raises(ValueError, match="every active completed image package"):
        protocol.check_geo_reference_packages(config, {}, tmp_path)


def test_direct_workflow_has_separate_identity_and_preserves_paired_randomness(config):
    assert protocol.study_root(config).name == protocol.DIRECT_STUDY_ID
    assert config["study_id"] not in (protocol.STRICT_STUDY_ID, protocol.EXPLORATORY_STUDY_ID)
    assert config["initialization_seed_namespace"] == "attention_fusion_v1"
    for stream in ("augmentation", "shuffle", "dropout"):
        assert config["training"][f"{stream}_seed_namespace"] == protocol.STRICT_STUDY_ID
    assert "multimodal/attention_fusion_joint_numerics.py" not in protocol.SOURCE_FILES
    assert "configs/multimodal_attention_fusion_joint_exploratory.yaml" not in protocol.SOURCE_FILES


@pytest.mark.parametrize("identity", [protocol.STRICT_STUDY_ID, protocol.EXPLORATORY_STUDY_ID])
def test_archived_workflows_cannot_be_executed_as_the_active_study(config, tmp_path, identity):
    config["study_id"] = identity
    path = tmp_path / "archived.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Archived M3J workflow"):
        protocol.load_config(path)


def test_archived_exploratory_config_is_retained_but_not_active():
    assert protocol.EXPLORATORY_CONFIG.is_file()
    with pytest.raises(ValueError, match="Archived M3J workflow"):
        protocol.load_config(protocol.EXPLORATORY_CONFIG)


@pytest.mark.parametrize("change", ["tolerance", "assurance", "root", "dropout", "replay", "resume", "verification"])
def test_direct_policy_cannot_silently_reintroduce_removed_gates(config, tmp_path, change):
    if change == "tolerance":
        config["numerical_policy"] = {"mode": "exploratory"}
    elif change == "assurance":
        config.pop("result_assurance")
    elif change == "root":
        config["paths"]["artifact_root"] = f"multimodal_artifacts/attention_fusion/cs/gse_100m/{protocol.STRICT_STUDY_ID}"
    elif change == "dropout":
        config["training"].pop("dropout_seed_namespace")
    elif change == "replay":
        config["checkpoint"]["prediction_replay"] = "exact_on_recorded_runtime"
    elif change == "resume":
        config["checkpoint"]["automatic_resume"] = True
    else:
        config["verification_policy"] = "exact_resume"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        protocol.load_config(path)


@pytest.mark.parametrize("field,value", [
    ("epochs", 5), ("batch_size", 128), ("gradient_accumulation_steps", 8),
    ("optimizer", "adamw"), ("learning_rate", 1e-4), ("scheduler_step", "per_batch"),
    ("autocast_dtype", "bfloat16"), ("grad_scaler", True), ("validation_autocast", True),
    ("matmul_allow_tf32", True), ("weight_decay", 1e-4),
])
def test_loading_refuses_silent_recipe_changes(config, tmp_path, field, value):
    config["training"][field] = value
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Training recipe differs"):
        protocol.load_config(path)


def test_full_matrix_is_seed_first_with_unique_geo_reference_and_exact_budget(config):
    fits = protocol.build_matrix(config)
    assert len(fits) == len({fit["run_id"] for fit in fits}) == 132
    assert Counter(fit["regime"] for fit in fits) == {
        "joint_partial": 64, "frozen_online": 64, "geo_only": 4}
    assert Counter(fit["initialization"] for fit in fits) == {
        "pretrained": 64, "cs_adapted": 64, "not_applicable": 4}
    for seed in range(1, 5):
        package = fits[(seed - 1) * 33:seed * 33]
        assert {fit["seed"] for fit in package} == {seed}
        assert [fit["regime"] for fit in package] == ["joint_partial"] * 16 + ["frozen_online"] * 16 + ["geo_only"]
        assert package[-1]["candidate_id"] == "geo_only"
        assert package[-1]["initialization"] == "not_applicable"
        for regime in ("joint_partial", "frozen_online"):
            arms = [fit for fit in package if fit["regime"] == regime]
            assert [(fit["candidate_id"], fit["initialization"]) for fit in arms] == [
                (candidate, initialization) for candidate in protocol.CANDIDATES
                for initialization in ("pretrained", "cs_adapted")]


def test_current_authorization_is_only_sixteen_seed_one_partial_joint_fits(config):
    authorized = protocol.authorized_fits(config)
    assert len(authorized) == 16
    assert authorized == protocol.build_matrix(config)[:16]
    assert {fit["seed"] for fit in authorized} == {1}
    assert {fit["regime"] for fit in authorized} == {"joint_partial"}
    assert {fit["candidate_id"] for fit in authorized} == set(protocol.CANDIDATES)
    assert authorized[0]["run_id"] == "seed1/joint_partial/image_only/pretrained"
    assert authorized[-1]["run_id"] == "seed1/joint_partial/attention_cls_skip_original/cs_adapted"


def test_frozen_online_has_separate_package_and_preserves_matched_recipe(config):
    frozen = protocol.load_config(protocol.FROZEN_ONLINE_CONFIG)
    assert protocol.study_root(frozen) != protocol.study_root(config)
    assert protocol.package_name(config) == "seed1_joint_partial"
    assert protocol.package_name(frozen) == "seed1_frozen_online"
    assert "configs/multimodal_attention_fusion_joint_frozen_online.yaml" in protocol.SOURCE_FILES
    fits = protocol.authorized_fits(frozen)
    assert len(fits) == 16
    assert fits == protocol.build_matrix(frozen)[16:32]
    assert {(fit["seed"], fit["regime"]) for fit in fits} == {(1, "frozen_online")}
    # Package identity, scope and the read-only paired reference are the only differences.
    comparable = deepcopy(frozen)
    for field in ("study_id", "paths", "authorized_scope"):
        comparable[field] = config[field]
    comparable["reporting"].pop("paired_reference_root")
    assert comparable == config


@pytest.mark.parametrize("which", ["partial", "frozen"])
@pytest.mark.parametrize("change", ["regime", "seed", "initializations", "root", "reference"])
def test_packages_reject_mixed_scope_roots_and_references(tmp_path, which, change):
    config = protocol.load_config(protocol.DEFAULT_CONFIG if which == "partial"
                                  else protocol.FROZEN_ONLINE_CONFIG)
    if change == "regime":
        config["authorized_scope"]["regime"] = ("frozen_online" if which == "partial"
                                                 else "joint_partial")
    elif change == "seed":
        config["authorized_scope"]["seeds"] = [1, 2]
    elif change == "initializations":
        config["authorized_scope"]["initializations"] = ["pretrained"]
    elif change == "root":
        config["paths"]["artifact_root"] = "multimodal_artifacts/attention_fusion/shared"
    else:
        config["reporting"]["paired_reference_root"] = "../unmatched_reference"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        protocol.load_config(path)


def test_contrasts_keep_checkpoint_policies_separate_and_pair_correct_controls(config):
    comparisons = protocol.contrasts(config)
    assert len(comparisons) == len({configuration_hash(item) for item in comparisons}) == 208
    by_policy = {policy: [item for item in comparisons if item["policy"] == policy]
                 for policy in ("final", "best")}
    assert all(len(items) == 104 for items in by_policy.values())
    # Both policies use the same comparison family, not different winning controls.
    stripped = [{configuration_hash({key: value for key, value in item.items() if key != "policy"})
                 for item in by_policy[policy]} for policy in ("final", "best")]
    assert stripped[0] == stripped[1]
    for items in by_policy.values():
        assert Counter(item["kind"] for item in items) == {
            "architecture": 72, "initialization": 16, "encoder_regime": 16}
        primary = [item for item in items if item["priority"] == "primary"]
        assert len(primary) == 4
        assert all(item["candidate"] == "attention_two_token"
                   and item["reference"] == "projected_concat" for item in primary)
        assert {(item["regime"], item["initialization"]) for item in primary} == {
            (regime, initialization) for regime in config["regimes"]
            for initialization in config["initializations"]}
        initialization = [item for item in items if item["kind"] == "initialization"]
        assert all(item["treatment"] == "cs_adapted" and item["reference"] == "pretrained"
                   for item in initialization)
        encoder = [item for item in items if item["kind"] == "encoder_regime"]
        assert all(item["treatment"] == "joint_partial" and item["reference"] == "frozen_online"
                   for item in encoder)
        architecture = [item for item in items if item["kind"] == "architecture"]
        for regime in config["regimes"]:
            for initialization in config["initializations"]:
                pairs = {(item["candidate"], item["reference"]) for item in architecture
                         if item["regime"] == regime and item["initialization"] == initialization}
                assert ("attention_two_token", "tokenwise_capacity") in pairs
                assert ("attention_cls_skip_adapted", "attention_cls") in pairs
                assert ("attention_cls_skip_original", "attention_cls") in pairs
                for candidate in protocol.CANDIDATES:
                    for anchor in ("image_only", "raw_concat"):
                        if candidate != anchor:
                            assert (candidate, anchor) in pairs


@pytest.fixture
def frozen_fixture(tmp_path, config):
    repo = tmp_path / "repo"
    for name in protocol.SOURCE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Source snapshot fixture for {name}\n")
    config["paths"]["artifact_root"] = "artifacts/joint"
    provenance = {"source_role": "train", "validation_role": "validation",
                  "test_features_or_labels_loaded": False, "assignments_sha256": "a" * 64,
                  "role_rows": {"train": 3378, "validation": 822}}
    return repo, config, provenance


@pytest.fixture
def paired_package_fixture(frozen_fixture):
    repo, _, provenance = frozen_fixture
    config = protocol.load_config(protocol.FROZEN_ONLINE_CONFIG)
    provenance["study_id"] = protocol.FROZEN_ONLINE_STUDY_ID
    reference_config = protocol.load_config()
    reference_provenance = dict(provenance, study_id=protocol.DIRECT_STUDY_ID)
    root = repo / config["reporting"]["paired_reference_root"]
    records = {
        "protocol/protocol.json": dict(
            config=reference_config, config_sha256=configuration_hash(reference_config),
            input_provenance=reference_provenance),
        "orchestration/seed1_joint_partial/status.json": dict(
            study_id=protocol.DIRECT_STUDY_ID, authorized_fits=16, complete=16,
            queue_status="complete", unresolved=[], active_fit=None, error=None,
            updated_at="2026-10-20T21:39:24.439081+00:00"),
        "reports/seed1_joint_partial/completion_audit.json": dict(
            package="seed1/joint_partial", status="complete", completed_fits=16,
            training_epochs=800, checkpoint_policy_evaluations=32, unresolved=[],
            test_data_scored=False),
    }
    for name, value in records.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return repo, config, provenance, root


@pytest.fixture
def geo_package_fixture(frozen_fixture):
    repo, _, provenance = frozen_fixture
    config = protocol.load_config(protocol.GEO_ONLY_CONFIG)
    provenance = dict(provenance, study_id=protocol.GEO_ONLY_STUDY_ID)
    for config_path in (protocol.DEFAULT_CONFIG, protocol.FROZEN_ONLINE_CONFIG):
        reference = protocol.load_config(config_path)
        root = protocol.study_root(reference, repo)
        record = root / "protocol/protocol.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps(dict(config=reference,
            config_sha256=configuration_hash(reference),
            input_provenance=dict(provenance, study_id=reference["study_id"]))))
        status = root / "orchestration" / protocol.package_name(reference) / "status.json"
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps(dict(study_id=reference["study_id"], queue_status="complete",
            complete=16, authorized_fits=16, active_fit=None, error=None, unresolved=[],
            updated_at="2026-10-22T00:49:39+00:00")))
    return repo, config, provenance


def test_geo_declaration_binds_both_completed_packages_read_only(geo_package_fixture):
    repo, config, provenance = geo_package_fixture
    references = [repo / relative for relative in config["reporting"]["completed_reference_roots"]]
    before = {p: p.read_bytes() for root in references for p in root.rglob("*.json")}
    payload = protocol.declare(config, provenance, repo)
    assert len(payload["authorized_fits"]) == 1
    assert [x["regime"] for x in payload["completed_reference_packages"]] == ["joint_partial", "frozen_online"]
    assert payload["realized_geo_shuffle_namespace"] == "attention_fusion_joint_v1"
    assert "do not isolate trainability" in payload["comparison_limitation"]
    assert {p: p.read_bytes() for p in before} == before


@pytest.fixture(params=[2, 3, 4])
def deferred_geo_package_fixture(frozen_fixture, request):
    repo, _, provenance = frozen_fixture
    seed = request.param
    config = protocol.load_config(protocol.package_config_path(seed, "geo_only"))
    provenance = dict(provenance, study_id=config["study_id"])
    reference = protocol.load_config(protocol.package_config_path(seed, "joint_partial"))
    root = protocol.study_root(reference, repo)
    record = root / "protocol/protocol.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(dict(config=reference,
        config_sha256=configuration_hash(reference),
        input_provenance=dict(provenance, study_id=reference["study_id"]))))
    status = root / "orchestration" / protocol.package_name(reference) / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps(dict(study_id=reference["study_id"], queue_status="complete",
        complete=16, authorized_fits=16, active_fit=None, error=None, unresolved=[],
        updated_at="2026-10-25T05:07:00+00:00")))
    return repo, config, provenance, root


def test_geo_declaration_binds_active_only_and_does_not_read_deferred_package(
        deferred_geo_package_fixture, monkeypatch):
    repo, config, provenance, root = deferred_geo_package_fixture
    seed = config["authorized_scope"]["seeds"][0]
    frozen_config = protocol.load_config(protocol.package_config_path(seed, "frozen_online"))
    deferred_root = protocol.study_root(frozen_config, repo)
    # Even a pre-existing but unfinished deferred package must not be inspected.
    deferred_record = deferred_root / "protocol/protocol.json"
    deferred_record.parent.mkdir(parents=True)
    deferred_record.write_text("unfinished package; not valid JSON")
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    original_read = type(deferred_record).read_text

    def guarded_read(path, *args, **kwargs):
        assert not path.is_relative_to(deferred_root), "Deferred package was read"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(type(deferred_record), "read_text", guarded_read)
    payload = protocol.declare(config, provenance, repo)
    assert [record["regime"] for record in payload["completed_reference_packages"]] == ["joint_partial"]
    assert payload["deferred_reference_packages"] == [dict(
        artifact_root=frozen_config["paths"]["artifact_root"], regime="frozen_online",
        seed=seed, status="deferred", planned_fits=16)]
    assert len(payload["full_matrix"]) == 132 and len(payload["authorized_fits"]) == 1
    assert "comparison_limitation" not in payload
    assert {p: p.read_bytes() for p in before} == before
    assert protocol.declare(config, provenance, repo) == payload


@pytest.mark.parametrize("change", ["missing", "recipe", "inputs", "incomplete", "wrong_seed"])
def test_geo_deferral_preserves_active_reference_guards(deferred_geo_package_fixture, change):
    repo, config, provenance, root = deferred_geo_package_fixture
    path = root / "protocol/protocol.json"
    record = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    elif change == "recipe":
        record["config"]["training"]["learning_rate"] = 1e-4
        record["config_sha256"] = configuration_hash(record["config"])
    elif change == "inputs":
        record["input_provenance"]["assignments_sha256"] = "different"
    elif change == "wrong_seed":
        record["config"]["authorized_scope"]["seeds"] = [1]
        record["config_sha256"] = configuration_hash(record["config"])
    else:
        seed = config["authorized_scope"]["seeds"][0]
        status_path = root / "orchestration" / f"seed{seed}_joint_partial/status.json"
        status = json.loads(status_path.read_text())
        status["complete"] = 15
        status_path.write_text(json.dumps(status))
    if change != "missing":
        path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        protocol.declare(config, provenance, repo)
    assert not (protocol.study_root(config, repo) / "protocol").exists()


@pytest.mark.parametrize("change", ["recipe", "inputs", "incomplete"])
def test_geo_declaration_rejects_unmatched_references(geo_package_fixture, change):
    repo, config, provenance = geo_package_fixture
    root = repo / config["reporting"]["completed_reference_roots"][1]
    path = root / "protocol/protocol.json"
    record = json.loads(path.read_text())
    if change == "recipe":
        record["config"]["training"]["learning_rate"] = 1e-4
        record["config_sha256"] = configuration_hash(record["config"])
    elif change == "inputs":
        record["input_provenance"]["assignments_sha256"] = "changed"
    else:
        status_path = root / "orchestration/seed1_frozen_online/status.json"
        status = json.loads(status_path.read_text())
        status["complete"] = 15
        status_path.write_text(json.dumps(status))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        protocol.declare(config, provenance, repo)
    assert not (protocol.study_root(config, repo) / "protocol").exists()


def test_frozen_declaration_binds_completed_reference_without_touching_it(paired_package_fixture):
    repo, config, provenance, reference_root = paired_package_fixture
    before = {p: p.read_bytes() for p in reference_root.rglob("*.json")}
    payload = protocol.declare(config, provenance, repo)
    reference = payload["paired_reference_package"]
    assert reference["completed_fits"] == 16
    assert reference["regime"] == "joint_partial" and reference["seed"] == 1
    assert reference["completion_audit_sha256"] == sha256_file(
        reference_root / "reports/seed1_joint_partial/completion_audit.json")
    assert reference["matched_input_identity_sha256"] == configuration_hash(
        {key: value for key, value in provenance.items() if key != "study_id"})
    assert "no weight-file audit or replay" in reference["check_scope"]
    assert protocol.declare(config, provenance, repo) == payload
    assert {p: p.read_bytes() for p in reference_root.rglob("*.json")} == before
    assert not any(reference_root.rglob("*.safetensors"))


@pytest.mark.parametrize("change", [
    "recipe", "inputs", "rng", "config_hash", "scope", "incomplete", "unresolved", "audit",
])
def test_frozen_declaration_rejects_unmatched_or_incomplete_reference(paired_package_fixture, change):
    repo, config, provenance, reference_root = paired_package_fixture
    path = reference_root / "protocol/protocol.json"
    previous = json.loads(path.read_text())
    if change == "recipe":
        previous["config"]["training"]["epochs"] = 49
    elif change == "inputs":
        previous["input_provenance"]["assignments_sha256"] = "b" * 64
    elif change == "rng":
        previous["config"]["training"]["shuffle_seed_namespace"] = "unpaired"
    elif change == "scope":
        previous["config"]["authorized_scope"]["seeds"] = [2]
    elif change in ("incomplete", "unresolved"):
        status_path = reference_root / "orchestration/seed1_joint_partial/status.json"
        status = json.loads(status_path.read_text())
        if change == "incomplete":
            status["complete"] = 15
        else:
            status["unresolved"] = ["unresolved checkpoint"]
        status_path.write_text(json.dumps(status))
    elif change == "audit":
        audit_path = reference_root / "reports/seed1_joint_partial/completion_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["checkpoint_policy_evaluations"] = 31
        audit_path.write_text(json.dumps(audit))
    previous["config_sha256"] = ("bad" if change == "config_hash"
                                  else configuration_hash(previous["config"]))
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError):
        protocol.declare(config, provenance, repo)
    assert not (protocol.study_root(config, repo) / "protocol/protocol.json").exists()


def test_declaration_persists_replayable_source_and_data_identity(frozen_fixture):
    repo, config, provenance = frozen_fixture
    payload = protocol.declare(config, provenance, repo)
    directory = protocol.study_root(config, repo) / "protocol"
    assert json.loads((directory / "protocol.json").read_text()) == payload
    assert payload["config_sha256"] == configuration_hash(config)
    assert payload["input_provenance"] == provenance
    assert payload["boundary"] == {"test_data_loaded": False, "validation_used_for_gradients": False}
    assert len(payload["full_matrix"]) == 132 and len(payload["authorized_fits"]) == 16
    assert "four-seed mean acc1" in payload["ranking"]
    assert "separate final/best" in payload["checkpoint_reporting"]
    assert "provisional" in payload["first_seed_conclusion"]
    for name, digest in payload["sources"].items():
        frozen = directory / "sources" / name
        assert frozen.read_bytes() == (repo / name).read_bytes()
        assert sha256_file(frozen) == digest
    before = (directory / "protocol.json").read_bytes()
    assert protocol.declare(deepcopy(config), deepcopy(provenance), repo) == payload
    assert (directory / "protocol.json").read_bytes() == before


def test_direct_declaration_records_verification_and_assurance_without_tolerance(frozen_fixture):
    repo, config, provenance = frozen_fixture
    payload = protocol.declare(config, provenance, repo)
    assert payload["result_assurance"] == protocol.PROVISIONAL_ASSURANCE
    assert payload["verification_policy"] == protocol.VERIFICATION_POLICY
    assert "numerical_policy" not in payload
    assert "numerical_policy" not in payload["config"]
    assert payload["config"]["study_id"] == protocol.DIRECT_STUDY_ID


@pytest.mark.parametrize("change", ["source", "inputs", "config"])
def test_declared_identity_refuses_changed_live_source_input_or_config(frozen_fixture, change):
    repo, config, provenance = frozen_fixture
    payload = protocol.declare(config, provenance, repo)
    directory = protocol.study_root(config, repo) / "protocol"
    before = (directory / "protocol.json").read_bytes()
    if change == "source":
        (repo / protocol.SOURCE_FILES[0]).write_text("# Different training implementation\n")
    elif change == "inputs":
        provenance["assignments_sha256"] = "b" * 64
    else:
        config["training"]["cpu_threads"] += 1
    with pytest.raises(ValueError, match="Declared M3J protocol/input/source changed"):
        protocol.declare(config, provenance, repo)
    assert (directory / "protocol.json").read_bytes() == before
    assert sha256_file(directory / "sources" / protocol.SOURCE_FILES[0]) == payload["sources"][protocol.SOURCE_FILES[0]]


@pytest.mark.parametrize("change", ["corrupt", "missing"])
def test_declaration_rechecks_frozen_source_snapshot_integrity(frozen_fixture, change):
    repo, config, provenance = frozen_fixture
    protocol.declare(config, provenance, repo)
    snapshot = protocol.study_root(config, repo) / "protocol/sources" / protocol.SOURCE_FILES[0]
    if change == "corrupt":
        snapshot.write_text("# Corrupted frozen source\n")
    else:
        snapshot.unlink()
    with pytest.raises(ValueError, match="Frozen source snapshot is missing or changed"):
        protocol.declare(config, provenance, repo)


@pytest.mark.parametrize("root", ["../outside", "/tmp/outside", "artifacts/../../outside"])
def test_artifact_root_cannot_escape_the_repository(config, tmp_path, root):
    config["paths"]["artifact_root"] = root
    with pytest.raises(ValueError, match="Unsafe artifact root"):
        protocol.study_root(config, tmp_path)


@pytest.fixture
def recovery_fixture(frozen_fixture):
    repo, config, provenance = frozen_fixture
    training = "multimodal/attention_fusion_joint_training.py"
    (repo / training).write_bytes((protocol.REPO_ROOT / training).read_bytes())
    original = protocol.declare(config, provenance, repo)
    root = protocol.study_root(config, repo)
    checked = dict(status="passed", sources=original["sources"],
                   config_sha256=configuration_hash(config), input_provenance=provenance)
    storage = repo / "multimodal/attention_fusion_joint_storage.py"
    storage.write_text(storage.read_text() + "# Engineering-only recovery amendment\n")

    def evidence():
        path = root / "verification/d039_cpu_tests.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="passed", failed=0, passed=10,
                                       sources=protocol.source_hashes(repo))))

    evidence()
    return repo, config, provenance, checked, evidence


def test_recovery_amendment_preserves_original_protocol_and_accepts_new_sources(recovery_fixture):
    repo, config, provenance, checked, evidence = recovery_fixture
    root = protocol.study_root(config, repo)
    original = (root / "protocol/protocol.json").read_bytes()
    amendment = protocol.declare_recovery_amendment(config, provenance, checked, repo)
    assert amendment["changed_files"] == ["multimodal/attention_fusion_joint_storage.py"]
    assert amendment["checkpoint_policy"]["expected_digest_required"] is True
    assert protocol.read_recovery_amendment(config, provenance, repo) == amendment
    assert protocol.declare_recovery_amendment(config, provenance, checked, repo) == amendment
    protocol.declare(config, provenance, repo)
    assert (root / "protocol/protocol.json").read_bytes() == original


@pytest.mark.parametrize("change", ["config", "input", "architecture", "training", "tests"])
def test_recovery_amendment_rejects_changed_science_or_missing_evidence(recovery_fixture, change):
    repo, config, provenance, checked, evidence = recovery_fixture
    if change == "config":
        config["training"]["epochs"] = 49
    elif change == "input":
        provenance = dict(provenance, assignments_sha256="b" * 64)
    elif change == "architecture":
        (repo / "multimodal/attention_fusion.py").write_text("# Model changed\n")
    elif change == "training":
        path = repo / "multimodal/attention_fusion_joint_training.py"
        path.write_text(path.read_text().replace("optimizer.step()", "optimizer.step(); optimizer.step()"))
    else:
        path = protocol.study_root(config, repo) / "verification/d039_cpu_tests.json"
        path.write_text(json.dumps({"status": "failed", "failed": 1}))
    if change != "tests":
        evidence()
    with pytest.raises(ValueError, match="D039"):
        protocol.declare_recovery_amendment(config, provenance, checked, repo)


def test_recovery_amendment_cannot_be_reused_after_source_changes(recovery_fixture):
    repo, config, provenance, checked, evidence = recovery_fixture
    protocol.declare_recovery_amendment(config, provenance, checked, repo)
    path = repo / "multimodal/attention_fusion_joint_storage.py"
    path.write_text(path.read_text() + "# Unreviewed later change\n")
    with pytest.raises(ValueError, match="different code/config/inputs"):
        protocol.read_recovery_amendment(config, provenance, repo)


@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("regime", protocol.PACKAGE_REGIMES)
def test_later_seed_templates_preserve_recipe_and_authorize_only_their_package(seed, regime):
    later = protocol.load_config(protocol.package_config_path(seed, regime))
    legacy = protocol.load_config(protocol.package_config_path(1, regime))
    assert later["study_id"] == protocol.package_identity(seed, regime)
    assert protocol.package_name(later) == f"seed{seed}_{regime}"
    assert protocol.study_root(later) != protocol.study_root(legacy)
    assert later["authorized_scope"]["seeds"] == [seed]
    assert len(protocol.build_matrix(later)) == 132
    fits = protocol.authorized_fits(later)
    assert len(fits) == (1 if regime == "geo_only" else 16)
    assert {(fit["seed"], fit["regime"]) for fit in fits} == {(seed, regime)}
    assert later["checkpoint"] == legacy["checkpoint"]
    assert later["geo"] == legacy["geo"] and later["visual"] == legacy["visual"]
    assert later["training"].pop("loader_rng_policy") == protocol.DECLARED_LOADER_RNG_POLICY
    assert later["training"] == legacy["training"]
    assert str(protocol.package_config_path(seed, regime).relative_to(protocol.REPO_ROOT)) in protocol.SOURCE_FILES


@pytest.mark.parametrize("seed", [2, 3, 4])
@pytest.mark.parametrize("regime", protocol.PACKAGE_REGIMES)
@pytest.mark.parametrize("change", ["seed", "root", "rng", "reference"])
def test_later_seed_package_guard_rejects_cross_seed_or_unpaired_settings(tmp_path, seed, regime, change):
    config = protocol.load_config(protocol.package_config_path(seed, regime))
    if change == "seed":
        config["authorized_scope"]["seeds"] = [1]
    elif change == "root":
        config["paths"] = protocol.load_config(protocol.package_config_path(1, regime))["paths"]
    elif change == "rng":
        config["training"].pop("loader_rng_policy")
    elif regime == "geo_only":
        config["reporting"]["completed_reference_roots"] = protocol.load_config(
            protocol.GEO_ONLY_CONFIG)["reporting"]["completed_reference_roots"]
    else:
        config["reporting"]["paired_reference_root"] = protocol.load_config(
            protocol.FROZEN_ONLINE_CONFIG)["reporting"]["paired_reference_root"]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        protocol.load_config(path)


@pytest.mark.parametrize("seed", [2, 3, 4])
def test_later_seed_declaration_matches_same_seed_references_and_records_actual_rng(frozen_fixture, seed):
    repo, _, base_provenance = frozen_fixture
    before = {}
    for regime in protocol.PACKAGE_REGIMES:
        config = protocol.load_config(protocol.package_config_path(seed, regime))
        provenance = dict(base_provenance, study_id=config["study_id"])
        payload = protocol.declare(config, provenance, repo)
        namespaces = payload["realized_loader_rng"]
        assert namespaces["policy"] == protocol.DECLARED_LOADER_RNG_POLICY
        assert namespaces["shuffle_seed_namespace"] == protocol.STRICT_STUDY_ID
        assert namespaces["augmentation_seed_namespace"] == (
            None if regime == "geo_only" else protocol.STRICT_STUDY_ID)
        assert "comparison_limitation" not in payload
        assert "same-seed" in payload["random_stream_pairing"]
        if regime == "frozen_online":
            assert payload["paired_reference_package"]["seed"] == seed
        if regime == "geo_only":
            references = payload["completed_reference_packages"]
            assert {reference["seed"] for reference in references} == {seed}
            assert all(reference["realized_loader_rng"]["policy"] ==
                       protocol.DECLARED_LOADER_RNG_POLICY for reference in references)
            assert all(reference["realized_loader_seed_namespace"] == protocol.STRICT_STUDY_ID
                       for reference in references)
        assert {path: path.read_bytes() for path in before} == before
        assert protocol.declare(config, provenance, repo) == payload
        root = protocol.study_root(config, repo)
        if regime != "geo_only":
            status = root / "orchestration" / f"seed{seed}_{regime}" / "status.json"
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps(dict(
                study_id=config["study_id"], authorized_fits=16, complete=16,
                queue_status="complete", unresolved=[], active_fit=None, error=None,
                updated_at="fixture_completed_at")))
            audit = root / "reports" / f"seed{seed}_{regime}" / "completion_audit.json"
            audit.parent.mkdir(parents=True)
            audit.write_text(json.dumps(dict(
                package=f"seed{seed}/{regime}", status="complete", completed_fits=16,
                training_epochs=800, checkpoint_policy_evaluations=32,
                unresolved=[], test_data_scored=False)))
            before.update({path: path.read_bytes() for path in root.rglob("*.json")})


@pytest.mark.parametrize("regime", ["frozen_online", "geo_only"])
def test_later_seed_declaration_requires_its_predecessors_and_does_not_use_seed1(frozen_fixture, regime):
    repo, _, provenance = frozen_fixture
    config = protocol.load_config(protocol.package_config_path(2, regime))
    with pytest.raises(FileNotFoundError):
        protocol.declare(config, dict(provenance, study_id=config["study_id"]), repo)
    assert not protocol.study_root(config, repo).exists()


@pytest.mark.parametrize("regime", protocol.PACKAGE_REGIMES)
def test_legacy_seed1_cannot_silently_switch_loader_rng_policy(tmp_path, regime):
    config = protocol.load_config(protocol.package_config_path(1, regime))
    config["training"]["loader_rng_policy"] = protocol.DECLARED_LOADER_RNG_POLICY
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Seed 1 retains its historical loader streams"):
        protocol.load_config(path)


@pytest.mark.parametrize("seed,regime", [(0, "joint_partial"), (5, "joint_partial"),
                                        (True, "joint_partial"), (2, "all")])
def test_package_identity_rejects_unknown_seed_and_regime(seed, regime):
    with pytest.raises(ValueError, match="seed 1–4"):
        protocol.package_identity(seed, regime)
