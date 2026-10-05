"""Synthetic-only selected-router transfer and fail-closed lineage checks."""

from copy import deepcopy
import subprocess
import sys

import numpy as np
import pytest

from multimodal import geo_helpfulness_legacy_capacity_predictions as adapter
from multimodal import geo_helpfulness_legacy_router_common as legacy
from multimodal import geo_helpfulness_router_capacity_state as capacity
from multimodal import geo_helpfulness_router_mlp_state as mlp
from test_multimodal_geo_legacy_router_common import frozen
from test_multimodal_geo_router_capacity_state import _state as tree_state, _xgb_leaf
from test_multimodal_geo_router_mlp_state import _state as neural_state


def _seal(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    value[key] = legacy.canonical_hash(value)
    return value


def _write(path, value, key=None):
    if key:
        value = _seal(value, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(legacy.canonical_bytes(value) + b"\n")
    return value


def _snapshot(root):
    return {
        str(path): legacy.file_hash(path) for path in root.rglob("*") if path.is_file()
    }


def _manifest(root, schema, declaration, files, **extra):
    return _write(
        root / "manifest.json",
        {
            "schema_version": schema,
            "declaration_sha256": declaration,
            "artifacts": {
                name: {
                    "file_sha256": legacy.file_hash(root / name),
                    "bytes": (root / name).stat().st_size,
                }
                for name in files
            },
            **extra,
        },
        "manifest_sha256",
    )


def _configs(family):
    rows = []
    learners = ("lr", "rf", "xgb") if family == "capacity" else ("mlp",)
    for learner in learners:
        for variant in adapter.VARIANTS:
            for index in range(4):
                parameters = {"C": [0.01, 0.1, 1.0, 10.0][index]}
                if learner == "rf":
                    parameters = {"n_estimators": 1}
                elif learner == "xgb":
                    parameters = {"n_estimators": 1, "base_score": 0.5}
                elif learner == "mlp":
                    parameters = {
                        **mlp.MLP_SHARED,
                        "hidden_widths": list(mlp.ARCHITECTURES[index]),
                    }
                rows.append(
                    {
                        "config_id": f"{learner}__{variant}__{index}",
                        "family": learner,
                        "variant": variant,
                        "parameters": parameters,
                    }
                )
    return {
        "study_id": f"router_{family}_v1",
        "schema_version": f"geo_helpfulness.router_{family}_config.v1",
        "configurations": rows,
    }


def _portable(spec, seed):
    family, variant = spec["family"], spec["variant"]
    if family == "mlp":
        state = neural_state(variant, tuple(spec["parameters"]["hidden_widths"]))
        state.update(initialization_seed=seed, shuffle_seed=seed, dropout_seed=seed)
    elif family == "xgb":
        state = tree_state(
            "xgb",
            variant,
            trees=[_xgb_leaf(value) for value in [0.6 + seed * 0.01, -0.3, 0.1, -0.1]],
        )
    else:
        state = tree_state("rf", variant)
    parameters = dict(spec["parameters"])
    if family != "mlp":
        parameters["random_state"] = seed
    state.update(
        config_id=spec["config_id"],
        training_seed=seed,
        fit_row_count=3378,
        parameters=parameters,
    )
    return (mlp if family == "mlp" else capacity).seal_state(state)


def _prediction_bundle(root, declaration, config, *, neural=False):
    configs = {row["config_id"]: row for row in config["configurations"]}
    files = adapter._prediction_files(configs, mlp=neural)
    selected = {
        row["config_id"]: row
        for row in config["configurations"]
        if row["config_id"].endswith("__0")
    }
    for name in files:
        legacy.write_bytes(
            root / "predictions" / name, b"synthetic opaque artifact; never decode\n"
        )
    for spec in selected.values():
        if spec["family"] == "lr":
            continue
        for seed in adapter.SEEDS:
            _write(
                root / "predictions/models" / f"{spec['config_id']}__seed_{seed}.json",
                {
                    "declaration_sha256": declaration["declaration_sha256"],
                    "state": _portable(spec, seed),
                },
                "record_sha256",
            )
    kind = "mlp" if neural else "capacity"
    return _manifest(
        root / "predictions",
        f"geo_helpfulness.router_{kind}_prediction_manifest.v1",
        declaration["declaration_sha256"],
        files,
    )


def _score_bundle(
    output,
    source,
    declaration,
    prediction,
    config,
    *,
    recovery=None,
    capacity_score=None,
):
    groups, selections = {}, {}
    for spec in config["configurations"]:
        if not spec["config_id"].endswith("__0"):
            continue
        group = spec["family"] + "__" + spec["variant"]
        threshold = deepcopy(
            legacy.NEVER if spec["variant"] == "ablation_no_pairs79" else legacy.ZERO
        )
        models = {}
        for seed in adapter.SEEDS:
            path = (
                source / "predictions/models" / f"{spec['config_id']}__seed_{seed}.json"
            )
            models[str(seed)] = {
                "path": str(path) if recovery else f"../predictions/models/{path.name}",
                "file_sha256": legacy.file_hash(path),
            }
        groups[group] = {
            "config_id": spec["config_id"],
            "threshold_specification": threshold,
            "models": models,
        }
        selections[group] = {
            "selected_config_id": spec["config_id"],
            "threshold": {"threshold_specification": threshold},
        }
    selection = {"groups": selections}
    frozen = {
        "study_id": declaration["study_id"],
        "groups": groups,
        "further_selection_or_refitting_permitted": False,
        "declaration_sha256" if recovery else "parent_declaration_sha256": declaration[
            "declaration_sha256"
        ],
    }
    if recovery:
        lineage = {
            "recovery_id": "router_mlp_v1_replay_v2",
            "recovery_declaration_sha256": recovery["declaration_sha256"],
            "source_declaration_sha256": declaration["declaration_sha256"],
            "source_prediction_manifest_sha256": prediction["manifest_sha256"],
            "source_files_changed": False,
        }
        frozen.update(
            recovery_lineage=lineage, model_path_basis="absolute_unchanged_source"
        )
    receipt = {
        "study_id": declaration["study_id"],
        "declaration_sha256": declaration["declaration_sha256"],
        "prediction_manifest_sha256": prediction["manifest_sha256"],
        "selection_sha256": legacy.canonical_hash(selection),
        "frozen_policies_sha256": legacy.canonical_hash(frozen),
        "post_receipt_reselection_permitted": False,
    }
    if recovery:
        receipt.update(
            recovery_lineage=lineage,
            capacity_score_manifest_sha256=capacity_score["manifest_sha256"],
        )
    files = adapter.MLP_SCORE_FILES if recovery else adapter.SCORE_FILES
    for name in files:
        legacy.write_bytes(
            output / "scores" / name, b"opaque development score; never decode\n"
        )
    _write(output / "scores/selection.json", selection)
    _write(output / "scores/frozen_policies.json", frozen)
    receipt = _write(
        output / "scores/selection_receipt.json", receipt, "receipt_sha256"
    )
    extra = {
        "prediction_manifest_sha256": prediction["manifest_sha256"],
        "selection_receipt_sha256": receipt["receipt_sha256"],
    }
    if recovery:
        extra["source_declaration_sha256"] = declaration["declaration_sha256"]
    return _manifest(
        output / "scores",
        (
            "geo_helpfulness.router_mlp_replay_score_manifest.v2"
            if recovery
            else "geo_helpfulness.router_capacity_score_manifest.v1"
        ),
        (
            recovery["declaration_sha256"]
            if recovery
            else declaration["declaration_sha256"]
        ),
        files,
        **extra,
    )


@pytest.fixture
def bundle(frozen, monkeypatch):
    # Legacy fixture imports development feature builders; isolate the real
    # fit-import denial check in a pristine subprocess below.
    monkeypatch.setattr(adapter, "_fit_incapable", lambda: None)
    root = frozen["root"]
    m4 = legacy.read_json(root / "router_frozen_state.json")
    for name, component in m4["components"].items():
        component["source"] = str(root / name)
    for variant, selection in m4["variants"].items():
        selection["selected_C_hex"] = (0.01).hex()
        selection["threshold_specification"] = (
            legacy.NEVER if variant == "primary_full727" else legacy.ZERO
        )
        for selected in selection["seeds"].values():
            selected["model_file_sha256"] = legacy.file_hash(
                root / selected["model_file"]
            )
            selected.update(_seal(selected, "frozen_state_sha256"))
    _write(root / "router_frozen_state.json", m4, "frozen_bundle_sha256")
    m4_manifest = _write(
        root / "manifest.json",
        {
            "schema_version": "geo_helpfulness.m4_feasibility_manifest.v1",
            "primary_decision": "no_go",
            "artifacts": {
                path.name: {"file_sha256": legacy.file_hash(path)}
                for path in root.iterdir()
                if path.is_file()
            },
        },
        "manifest_sha256",
    )
    declaration = dict(frozen["declaration"], source_snapshot=_snapshot(root))
    declaration = _write(
        frozen["output"] / "declaration.json", declaration, "declaration_sha256"
    )
    frozen["declaration"] = declaration
    capacity_root, mlp_root, replay_root = (
        root.parent / name for name in ("capacity", "mlp", "replay")
    )
    cap_config, mlp_config = _configs("capacity"), _configs("mlp")
    cap = _write(
        capacity_root / "declaration.json",
        {
            "schema_version": "geo_helpfulness.router_capacity_declaration.v1",
            "study_id": "router_capacity_v1",
            "configuration": cap_config,
            "protocol_root": str(root.parent.parent),
            "parent_file_sha256": _snapshot(root),
            "original_m4_manifest_sha256": m4_manifest["manifest_sha256"],
        },
        "declaration_sha256",
    )
    cap_prediction = _prediction_bundle(capacity_root, cap, cap_config)
    cap_score = _score_bundle(
        capacity_root, capacity_root, cap, cap_prediction, cap_config
    )
    neural = _write(
        mlp_root / "declaration.json",
        {
            "schema_version": "geo_helpfulness.router_mlp_declaration.v1",
            "study_id": "router_mlp_v1",
            "configuration": mlp_config,
            "protocol_root": str(root.parent.parent),
            "parent_file_sha256": {**_snapshot(root), **_snapshot(capacity_root)},
            "original_m4_manifest_sha256": m4_manifest["manifest_sha256"],
            "capacity_root": str(capacity_root),
            "capacity_prediction_manifest_sha256": cap_prediction["manifest_sha256"],
            "capacity_score_manifest_sha256": cap_score["manifest_sha256"],
        },
        "declaration_sha256",
    )
    mlp_prediction = _prediction_bundle(mlp_root, neural, mlp_config, neural=True)
    recovery = _write(
        replay_root / "declaration.json",
        {
            "schema_version": "geo_helpfulness.router_mlp_replay_declaration.v2",
            "study_id": "router_mlp_v1",
            "recovery_id": "router_mlp_v1_replay_v2",
            "source_root": str(mlp_root),
            "source_declaration_sha256": neural["declaration_sha256"],
            "source_prediction_manifest_sha256": mlp_prediction["manifest_sha256"],
            "source_mutations_permitted": False,
            "fitting_permitted": False,
            "scientific_selection_rules_changed": False,
            "parent_file_sha256": {
                **neural["parent_file_sha256"],
                **_snapshot(mlp_root),
            },
        },
        "declaration_sha256",
    )
    _score_bundle(
        replay_root,
        mlp_root,
        neural,
        mlp_prediction,
        mlp_config,
        recovery=recovery,
        capacity_score=cap_score,
    )
    return {**frozen, "capacity": capacity_root, "mlp": mlp_root, "replay": replay_root}


def _load(bundle):
    return adapter.load_selected_policies(
        bundle["capacity"], bundle["mlp"], bundle["replay"], bundle["declaration"]
    )


def _frames(bundle):
    return {
        name: table.to_pandas()
        for name, table in legacy.reconstruct_predictions(
            bundle["logits"], bundle["declaration"]
        ).items()
    }


def test_selected_load_authenticates_all36_models_without_parquet_or_metrics_decoding(
    bundle, monkeypatch
):
    import pyarrow.parquet as pq

    monkeypatch.setattr(
        pq,
        "read_table",
        lambda *a, **k: pytest.fail("selection loading decoded a table"),
    )
    before = {
        str(path): (legacy.file_hash(path), path.stat().st_mtime_ns)
        for path in bundle["root"].parent.rglob("*")
        if path.is_file()
    }
    selection = _load(bundle)
    assert list(selection["policies"]) == list(adapter.POLICIES)
    assert set(selection["states"]) == set(adapter.NEW_POLICIES)
    assert sum(map(len, selection["states"].values())) == 36
    assert selection["lineage"]["original_m4_no_go_preserved"] is True
    assert _load(bundle) == selection
    assert before == {
        str(path): (legacy.file_hash(path), path.stat().st_mtime_ns)
        for path in bundle["root"].parent.rglob("*")
        if path.is_file()
    }


def test_reconstruct_full_crossproduct_preserves_experts_and_all_old_values(bundle):
    selection, frames = _load(bundle), _frames(bundle)
    tables = adapter.reconstruct_predictions(frames, selection, bundle["declaration"])
    count = len(frames["experts"])
    assert tables["experts"].num_columns == 46
    assert tables["routers"].schema == adapter.ROUTER_SCHEMA
    assert tables["routers"].num_rows == count * 12
    assert tables["actions"].num_rows == count * 17
    assert all(
        not field.nullable for table in tables.values() for field in table.schema
    )
    assert tables["experts"].equals(
        legacy._table(frames["experts"], "experts", bundle["feature_schema"])
    )
    actions, routers = tables["actions"].to_pandas(), tables["routers"].to_pandas()
    old = actions.loc[actions.policy_id.isin(legacy.POLICIES)].reset_index(drop=True)
    assert old.equals(frames["actions"])
    assert set(routers.study_id) == {adapter.STUDY_ID}
    for variant in adapter.VARIANTS:
        saved = (
            frames["routers"]
            .loc[frames["routers"].variant_id.eq(variant)]
            .reset_index(drop=True)
        )
        observed = routers.loc[routers.policy_id.eq(variant)].reset_index(drop=True)
        assert observed[[*adapter.Q_COLUMNS, "score", "C_hex"]].equals(
            saved[[*adapter.Q_COLUMNS, "score", "C_hex"]]
        )
    for policy_id in adapter.NEW_POLICIES:
        policy = selection["policies"][policy_id]
        subset = actions.loc[actions.policy_id.eq(policy_id)]
        if policy["threshold_specification"] == legacy.NEVER:
            assert not subset.acted.any()
        assert not subset.loc[subset.raw_pred.eq(subset.geo_pred), "acted"].any()
        assert np.array_equal(
            subset.final_pred, np.where(subset.acted, subset.geo_pred, subset.raw_pred)
        )
    replayed = adapter.reconstruct_predictions(frames, selection, bundle["declaration"])
    assert all(tables[name].equals(replayed[name]) for name in tables)


@pytest.mark.parametrize("learner", ["rf", "xgb", "mlp"])
@pytest.mark.parametrize("variant", list(adapter.VARIANTS))
def test_each_selected_predictor_receives_matching_family_seed_transform(
    bundle, learner, variant
):
    selection, frames = _load(bundle), _frames(bundle)
    tables = adapter.reconstruct_predictions(frames, selection, bundle["declaration"])
    policy = ("" if learner == "mlp" else "capacity__") + learner + "__" + variant
    routers = tables["routers"].to_pandas()
    for family in legacy.FAMILIES:
        for seed in adapter.SEEDS:
            experts = (
                frames["experts"]
                .loc[
                    frames["experts"].checkpoint_family.eq(family)
                    & frames["experts"].training_seed.eq(seed)
                ]
                .reset_index(drop=True)
            )
            matrix = legacy.apply_transform(experts, bundle["transforms"][seed - 1])
            expected = (mlp if learner == "mlp" else capacity).predict_state(
                matrix, selection["states"][policy][str(seed)]
            )
            observed = routers.loc[
                routers.policy_id.eq(policy)
                & routers.checkpoint_family.eq(family)
                & routers.training_seed.eq(seed)
            ]
            assert np.array_equal(
                observed[list(adapter.Q_COLUMNS)].to_numpy(), expected
            )


@pytest.mark.parametrize("part", ["experts", "routers", "actions"])
def test_label_or_unknown_columns_rejected_before_replay(bundle, part):
    selection, frames = _load(bundle), _frames(bundle)
    frames[part]["label_id_dense"] = 0
    with pytest.raises(ValueError, match="exact label-blind"):
        adapter.reconstruct_predictions(frames, selection, bundle["declaration"])


@pytest.mark.parametrize(
    "problem",
    [
        "missing_seed",
        "duplicate_identity",
        "wrong_family",
        "wrong_feature",
        "old_action",
        "old_q",
        "state_seed",
        "state_resealed",
    ],
)
def test_reconstruction_rejects_identity_or_prediction_or_selected_state_tamper(
    bundle, problem
):
    selection, frames = _load(bundle), _frames(bundle)
    if problem == "missing_seed":
        frames["experts"] = (
            frames["experts"]
            .loc[frames["experts"].training_seed.ne(4)]
            .reset_index(drop=True)
        )
    elif problem == "duplicate_identity":
        frames["experts"].loc[4, "row_uid"] = frames["experts"].loc[0, "row_uid"]
    elif problem == "wrong_family":
        frames["experts"].loc[0, "checkpoint_family"] = "unknown"
    elif problem == "wrong_feature":
        frames["experts"].loc[0, "raw_confidence"] += 0.1
    elif problem == "old_action":
        frames["actions"].loc[0, "acted"] = not frames["actions"].loc[0, "acted"]
    elif problem == "old_q":
        frames["routers"].loc[0, "q_rescue"] += 0.1
    else:
        key = adapter.NEW_POLICIES[0]
        state = selection["states"][key]["1"]
        if problem == "state_seed":
            state["training_seed"] = 2
        else:
            state["model"]["trees"][0]["threshold"][0] += 0.01
            selection["states"][key]["1"] = capacity.seal_state(state)
    with pytest.raises(ValueError):
        adapter.reconstruct_predictions(frames, selection, bundle["declaration"])


@pytest.mark.parametrize(
    "problem",
    [
        "parent",
        "manifest",
        "receipt",
        "model",
        "unknown_file",
        "symlink",
        "recovery",
        "preprocessing",
    ],
)
def test_parent_bundle_loading_fails_closed(bundle, problem):
    if problem == "parent":
        (bundle["root"] / "expert_temperatures.json").chmod(0o600)
        (bundle["root"] / "expert_temperatures.json").write_bytes(b"changed")
    elif problem == "unknown_file":
        legacy.write_bytes(bundle["capacity"] / "scores/unowned.json", b"{}")
    elif problem == "symlink":
        (bundle["replay"] / "link").symlink_to(bundle["mlp"])
    else:
        path = {
            "manifest": bundle["capacity"] / "predictions/manifest.json",
            "receipt": bundle["replay"] / "scores/selection_receipt.json",
            "model": bundle["mlp"]
            / "predictions/models/mlp__primary_full727__0__seed_1.json",
            "recovery": bundle["replay"] / "declaration.json",
            "preprocessing": bundle["capacity"] / "declaration.json",
        }[problem]
        value = legacy.read_json(path)
        if problem == "recovery":
            value["source_root"] = str(bundle["capacity"])
            _write(path, value, "declaration_sha256")
        elif problem == "preprocessing":
            value["parent_file_sha256"][
                str(bundle["root"] / "expert_temperatures.json")
            ] = ("f" * 64)
            _write(path, value, "declaration_sha256")
        else:
            value["tampered"] = True
            _write(path, value)
    with pytest.raises(ValueError):
        _load(bundle)


def test_threshold_equality_and_geo_raw_agreement_are_never_actions(
    bundle, monkeypatch
):
    selection, frames = _load(bundle), _frames(bundle)
    monkeypatch.setattr(
        capacity,
        "predict_state",
        lambda matrix, state: np.tile([0.5, 0.25, 0.125, 0.125], (len(matrix), 1)),
    )
    target = adapter.NEW_POLICIES[0]
    selection["policies"][target]["threshold_specification"] = {
        "kind": "finite",
        "comparison": "strict_gt",
        "value_hex": (0.25).hex(),
    }
    result = adapter.reconstruct_predictions(frames, selection, bundle["declaration"])[
        "actions"
    ].to_pandas()
    assert not result.loc[result.policy_id.eq(target)].acted.any()


def test_module_import_and_guard_are_fit_incapable():
    source = """
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','scipy','sklearn','xgboost'} or fullname.endswith(('_fit','_inputs','_selection')):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Deny())
from multimodal import geo_helpfulness_legacy_capacity_predictions as adapter
adapter._fit_incapable()
sys.modules['torch'] = object()
try:
    adapter._fit_incapable()
except ValueError:
    pass
else:
    raise AssertionError('fit guard did not reject torch')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", source], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
