from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import multimodal.geo_helpfulness_protocol as protocol
from multimodal.geo_helpfulness_locked_eval import canonical_json_bytes as locked_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "multimodal_geo_helpfulness.yaml"
RUNNER_PATH = REPO_ROOT / "tools" / "run_multimodal_geo_helpfulness.py"

EXPECTED_ONTOLOGY = (
    (0, 0, "Urban"),
    (1, 1, "Broadleaved Mixed and Yew Woodland"),
    (2, 2, "Coniferous Woodland"),
    (3, 4, "Arable and Horticulture"),
    (4, 5, "Improved Grassland"),
    (5, 6, "Neutral Grassland"),
    (6, 7, "Calcareous Grassland"),
    (7, 8, "Acid Grassland"),
    (8, 9, "Bracken"),
    (9, 10, "Dwarf Shrub Heath"),
    (10, 11, "Fen, Marsh, Swamp"),
    (11, 12, "Bog"),
    (12, 13, "Littoral Rock"),
    (13, 14, "Littoral Sediment"),
    (14, 15, "Montane"),
    (15, 17, "Inland Rock"),
    (16, 18, "Supra-littoral Rock"),
    (17, 19, "Supra-littoral Sediment"),
)


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(path: Path, config: dict) -> Path:
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _identity_frame(prefix: str, plot_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": [f"images/{prefix}_{idx}.jpg" for idx in range(len(plot_ids))],
            "plot_idx": plot_ids,
        }
    )


def _development_fixture() -> pd.DataFrame:
    """Small plot-labelled fixture with one singleton and all 18 head classes."""

    rows: list[dict] = []
    for dense_id, _canonical_id, label_name in EXPECTED_ONTOLOGY:
        # Class 12 is deliberately a singleton.  Other classes have enough plots
        # for both roles; repeated photos exercise plot-level grouping.
        plot_count = 1 if dense_id == 12 else 5
        for plot_number in range(plot_count):
            plot_idx = f"C{dense_id:02d}X{plot_number + 1}"
            image_count = 2 if plot_number == 0 else 1
            for image_number in range(image_count):
                rows.append(
                    {
                        "file": f"habitat_{dense_id:02d}/{plot_idx}_{image_number}.jpg",
                        "plot_idx": plot_idx,
                        "label_id": dense_id,
                        "label_id_original": _canonical_id,
                        "label_name": label_name,
                        "image_source": "synthetic_pretrained",
                        "split": "train" if plot_number % 2 == 0 else "val",
                    }
                )
    return pd.DataFrame(rows)


def _call_build_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the fixture constants explicit at the protocol API boundary."""

    return protocol.build_development_assignments(
        frame,
        **_assignment_kwargs(frame),
    )


def _assignment_kwargs(frame: pd.DataFrame) -> dict:
    return {
        "protocol_id": "synthetic_protocol_v1",
        "dataset_id": "synthetic_cs",
        "role_seed": 20260824,
        "validation_plot_count": 17,
        "n_oof_folds": 4,
        "oof_seed": 20261824,
        "expected_rows": len(frame),
        "expected_plots": int(frame["plot_idx"].nunique()),
    }


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER_PATH), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_cli_protocol_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = _load_config()
    development = _development_fixture()
    image_root = tmp_path / "development_images"
    image_root.mkdir()
    development["image_source"] = str(image_root)
    for file_name in development["file"]:
        image_path = image_root / str(file_name)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"synthetic image: {file_name}\n".encode("utf-8"))
    train_path = tmp_path / "development_train.parquet"
    val_path = tmp_path / "development_val.parquet"
    development[development["split"] == "train"].to_parquet(train_path, index=False)
    development[development["split"] == "val"].to_parquet(val_path, index=False)

    test_source = _identity_frame("sealed_test", ["LOCKED1", "LOCKED2", "LOCKED3"])
    # The sealing command must project these forbidden columns away.
    test_source["label_id"] = [0, 1, 2]
    test_source["A00"] = [0.1, 0.2, 0.3]
    test_path = tmp_path / "cleaned_test_source.parquet"
    test_source.to_parquet(test_path, index=False)

    plot_count = int(development["plot_idx"].nunique())
    registry = tmp_path / "locked_registry"
    protocol_root = tmp_path / "protocol"
    config["protocol_status"] = "frozen"
    config["experts"]["image_encoder"]["strategy"] = "externally_pretrained_fixed"
    config["experts"]["learned_state"]["fold_local"] = [
        "image_only_head",
        "geo_only_head",
        "raw_concat_head",
        "geo_standardization",
    ]
    config["experts"]["learned_state"]["expert_refit_state"] = list(
        config["experts"]["learned_state"]["fold_local"]
    )
    config["paths"]["development_source_tables"] = [str(train_path), str(val_path)]
    config["paths"]["locked_test_registry_root"] = str(registry)
    config["paths"]["protocol_root"] = str(protocol_root)
    config["paths"]["global_test_event_registry"] = str(registry / "events")
    config["development_universe"]["expected_rows"] = int(len(development))
    config["development_universe"]["expected_plots"] = plot_count
    config["development_universe"]["allowed_image_sources"] = [str(image_root)]
    config["locked_test_identity"]["expected_rows"] = int(len(test_source))
    config["locked_test_identity"]["expected_plots"] = int(
        test_source["plot_idx"].nunique()
    )
    config["assignment"]["role"]["exact_validation_plots"] = 17
    config["assignment"]["role"]["exact_train_plots"] = plot_count - 17
    config_path = tmp_path / "protocol_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, test_path, protocol_root


def test_frozen_config_is_self_contained_and_has_no_test_table_path():
    config = _load_config()

    assert config["schema_version"] == "geo_helpfulness_protocol_config_v1"
    assert config["protocol_id"] == "protocol_v1"
    assert "locked_test_identity_source" not in config["paths"]
    assert config["paths"]["locked_test_registry_root"].endswith(
        "locked_test_registry/cs/gse_100m_cleaned_test"
    )
    assert config["development_universe"]["expected_rows"] == 4200
    assert config["development_universe"]["expected_plots"] == 1625
    assert config["assignment"]["role"]["exact_train_plots"] == 1300
    assert config["assignment"]["role"]["exact_validation_plots"] == 325
    assert config["assignment"]["train_oof"]["n_splits"] == 4
    assert config["experts"]["training_seeds"] == [1, 2, 3, 4]
    assert config["experts"]["report_test_each_epoch"] is False
    assert config["experts"]["allow_oracle_test_best"] is False
    source_columns = config["development_universe"]["source_columns"]
    assert "dense_label" not in source_columns
    assert "canonical_label" not in source_columns
    assert config["class_ontology"]["development_label_resolution"] == (
        "exact_label_name_lookup_in_frozen_ontology"
    )

    calibration = config["calibration"]
    assert calibration["expert_probability"]["fit_role"] == "development_train_oof"
    assert calibration["expert_probability"]["fit_once"] is True
    assert calibration["router_output_calibrator"] == "none"
    assert config["router"]["regularization_grid_C"] == [0.01, 0.1, 1.0, 10.0]
    assert config["policy"]["effective_action"]["require_geo_raw_disagreement"] is True
    assert config["policy"]["threshold_candidates"]["allow_negative"] is False
    assert config["evidence"]["pass_rule"]["all_required"]
    assert config["evidence"]["no_go_if_any"]

    feature_names = {
        name
        for family in config["router"]["feature_allowlist"].values()
        for name in family
    }
    forbidden = ("true", "correct", "label", "target", "nll")
    assert not any(token in name for name in feature_names for token in forbidden)
    matrix = config["router"]["feature_matrix"]
    scaled_count = sum(
        len(config["router"]["feature_allowlist"][family])
        for family in ("boolean", "integer", "numeric")
    )
    assert scaled_count == matrix["scaled_numeric_column_count"] == 25
    assert matrix["one_hot_column_count"] == 3 * 18 + 2 * 18 * 18
    assert matrix["total_column_count"] == 727


def test_frozen_config_and_code_share_the_fixed_test_independent_ontology():
    config = _load_config()
    configured = tuple(
        (entry["dense_id"], entry["canonical_l3_id"], entry["label_name"])
        for entry in config["class_ontology"]["classes"]
    )

    assert configured == EXPECTED_ONTOLOGY
    assert tuple(protocol.FIXED_CLASS_ONTOLOGY) == EXPECTED_ONTOLOGY
    assert config["class_ontology"]["output_size"] == 18
    assert config["class_ontology"]["source"] == "frozen_protocol_not_data_inference"


def test_frozen_config_and_code_share_artifact_parent_role_allowlists():
    configured = _load_config()["artifact_contract"]["artifact_parent_roles"]

    for artifact_role, parent_roles in configured.items():
        protocol.validate_artifact_parent_roles(artifact_role, parent_roles)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("images\\plot.1.jpg", "images/plot.1.jpg"),
        ("images//./sub/plot.jpg", "images/sub/plot.jpg"),
        ("cafe\N{COMBINING ACUTE ACCENT}.jpg", "caf\N{LATIN SMALL LETTER E WITH ACUTE}.jpg"),
    ],
)
def test_canonicalize_file_is_stable(raw: str, expected: str):
    assert protocol.canonicalize_file(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "/absolute.jpg", "../escape.jpg", "images/../escape.jpg", "bad\x00.jpg"],
)
def test_canonicalize_file_rejects_unsafe_identities(raw: str):
    with pytest.raises((TypeError, ValueError)):
        protocol.canonicalize_file(raw)


def test_plot_identity_is_opaque_case_sensitive_text_and_rejects_whitespace():
    assert protocol.canonicalize_plot_idx("751X3") == "751X3"
    assert protocol.canonicalize_plot_idx("751x3") == "751x3"
    assert protocol.canonicalize_plot_idx("cafe\N{COMBINING ACUTE ACCENT}") == (
        "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    )
    for invalid in (751, "", " 751X3", "751X3 ", "bad\x00plot", "bad\nplot"):
        with pytest.raises((TypeError, ValueError)):
            protocol.canonicalize_plot_idx(invalid)


def test_row_uid_is_stable_after_allowed_file_normalization():
    first = protocol.make_row_uid("synthetic_cs", "Images\\A.JPG", "751X3")
    second = protocol.make_row_uid("synthetic_cs", "Images/A.JPG", "751X3")

    assert first == second
    assert len(first) == 64
    assert first == first.lower()
    assert first != protocol.make_row_uid("synthetic_cs", "Images/A.JPG", "751x3")


def test_all_protocol_surfaces_share_one_canonical_json_serialization():
    value = {"z": [1, 2.5], "accent": "cafe\N{COMBINING ACUTE ACCENT}"}

    assert locked_json_bytes(value) == protocol.canonical_json_bytes(value)


def test_identity_projection_is_order_independent_and_rejects_casefold_collision():
    frame = _identity_frame("sample", ["P2", "P1", "P3"])
    first = protocol.build_identity_projection(frame, dataset_id="synthetic_cs")
    second = protocol.build_identity_projection(
        frame.sample(frac=1.0, random_state=7).reset_index(drop=True),
        dataset_id="synthetic_cs",
    )

    pd.testing.assert_frame_equal(first, second)
    collision = pd.DataFrame(
        {"file": ["images/A.jpg", "images/a.JPG"], "plot_idx": ["P1", "P2"]}
    )
    with pytest.raises(ValueError, match="(?i)(collision|duplicate|unique)"):
        protocol.build_identity_projection(collision, dataset_id="synthetic_cs")


def test_locked_identity_manifest_is_label_blind_and_tamper_evident():
    source = _identity_frame("test", ["T1", "T2"])
    source["label_id"] = [3, 4]
    source["A00"] = [0.25, 0.75]
    manifest = protocol.build_locked_test_identity_manifest(
        source,
        dataset_id="synthetic_cs_test",
    )

    serialized = json.dumps(manifest, sort_keys=True)
    assert "label_id" not in serialized
    assert "A00" not in serialized
    protocol.validate_locked_test_identity_manifest(manifest)

    tampered = json.loads(json.dumps(manifest))
    tampered["rows"][0]["plot_idx"] = "MUTATED"
    with pytest.raises(ValueError, match="(?i)(hash|fingerprint|identity)"):
        protocol.validate_locked_test_identity_manifest(tampered)

    for field, value in (
        ("artifact_role", "training_data"),
        ("canonical_json_version", "different.v1"),
        ("identity_projection_columns", ["row_uid"]),
    ):
        invalid_contract = json.loads(json.dumps(manifest))
        invalid_contract[field] = value
        invalid_contract["manifest_sha256"] = protocol.canonical_sha256(
            {
                key: item
                for key, item in invalid_contract.items()
                if key != "manifest_sha256"
            }
        )
        with pytest.raises(ValueError, match="(?i)(role|version|columns|contract)"):
            protocol.validate_locked_test_identity_manifest(invalid_contract)


def test_assignments_are_deterministic_and_input_order_independent():
    frame = _development_fixture()
    first = _call_build_assignments(frame)
    second = _call_build_assignments(
        frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(first, second)
    assert protocol.content_sha256(first) == protocol.content_sha256(second)
    assert len(first) == len(frame)
    assert first["row_uid"].is_unique
    assert set(first["development_role"]) == {"train", "validation"}
    assert first.loc[first["development_role"] == "validation", "train_oof_fold"].isna().all()
    train_folds = first.loc[first["development_role"] == "train", "train_oof_fold"]
    assert set(train_folds.astype(int)) == {0, 1, 2, 3}


def test_assignment_keeps_plots_grouped_and_singleton_class_in_train():
    assignments = _call_build_assignments(_development_fixture())

    assert assignments.groupby("plot_idx")["development_role"].nunique().max() == 1
    train = assignments[assignments["development_role"] == "train"]
    assert train.groupby("plot_idx")["train_oof_fold"].nunique().max() == 1
    singleton = assignments[assignments["label_id_dense"] == 12]
    assert set(singleton["development_role"]) == {"train"}
    assert set(assignments["label_id_dense"]) == set(range(18))
    assert set(assignments["canonical_l3_id"]) == {item[1] for item in EXPECTED_ONTOLOGY}
    protocol.validate_development_assignments(
        assignments,
        **_assignment_kwargs(_development_fixture()),
    )


def test_development_universe_cannot_shrink_the_fixed_head():
    incomplete = _development_fixture()
    incomplete = incomplete[incomplete["label_id"] != 17].reset_index(drop=True)

    with pytest.raises(ValueError, match="complete frozen 18-class ontology"):
        protocol.build_development_assignments(
            incomplete,
            protocol_id="synthetic_protocol_v1",
            dataset_id="synthetic_cs",
            role_seed=20260824,
            validation_plot_count=16,
            n_folds=4,
            fold_seed=20261824,
        )


def test_overlap_validator_rejects_locked_test_rows_and_plots():
    assignments = _call_build_assignments(_development_fixture())
    overlap = assignments.iloc[[0]][["file", "plot_idx"]]
    locked = protocol.build_locked_test_identity_manifest(
        overlap,
        dataset_id="synthetic_cs",
    )

    with pytest.raises(ValueError, match="(?i)overlap"):
        protocol.validate_development_assignments(
            assignments,
            test_identity_manifest=locked,
            **_assignment_kwargs(_development_fixture()),
        )


def test_content_hash_and_cache_validation_reject_mutation():
    assignments = _call_build_assignments(_development_fixture())
    manifest = protocol.build_artifact_manifest(
        artifact_role="development_assignments",
        protocol_id="synthetic_protocol_v1",
        payload=assignments,
    )
    protocol.validate_artifact_manifest(manifest, payload=assignments)

    stale = assignments.copy()
    stale.loc[0, "development_role"] = (
        "validation" if stale.loc[0, "development_role"] == "train" else "train"
    )
    with pytest.raises(ValueError, match="(?i)(hash|fingerprint|stale|content)"):
        protocol.validate_artifact_manifest(manifest, payload=stale)


def test_artifact_parent_roles_reject_final_in_sample_router_training_parent():
    protocol.validate_artifact_parent_roles(
        "router_training_dataset",
        ["development_train_oof_outputs"],
    )
    with pytest.raises(ValueError, match="(?i)(parent|role)"):
        protocol.validate_artifact_parent_roles(
            "router_training_dataset",
            ["final_development_in_sample_outputs"],
        )


def test_prediction_provenance_requires_disjoint_fit_and_prediction_plots():
    provenance = protocol.validate_fit_prediction_plot_provenance(
        ["TRAIN1", "TRAIN2"],
        ["HELDOUT1", "HELDOUT2"],
    )
    assert provenance["zero_plot_overlap"] is True
    assert provenance["fitting_plot_count"] == 2
    assert provenance["prediction_plot_count"] == 2

    with pytest.raises(ValueError, match="plot overlap"):
        protocol.validate_fit_prediction_plot_provenance(
            ["TRAIN1", "SHARED"],
            ["SHARED", "HELDOUT"],
        )


def test_validation_outputs_accept_only_declared_development_train_producers():
    protocol.validate_artifact_parent_roles(
        "development_validation_outputs",
        ["development_assignments", "development_train_expert_fit"],
    )
    with pytest.raises(ValueError, match="forbidden parent"):
        protocol.validate_artifact_parent_roles(
            "development_validation_outputs",
            ["development_assignments", "full_development_expert_fit"],
        )


@pytest.mark.parametrize(
    "command",
    [
        "build-train-oof",
        "fit-router-candidates",
        "score-router-candidates",
        "fit-final-experts",
    ],
)
def test_post_m1_commands_fail_closed(command: str):
    result = _run_cli(command)

    assert result.returncode != 0
    assert "not implemented in M1" in (result.stdout + result.stderr)


def test_identity_source_is_command_local_and_required():
    result = _run_cli("freeze-test-identity", "--help")

    assert result.returncode == 0
    help_text = result.stdout + result.stderr
    assert "--source" in help_text
    assert "--file-column" not in help_text
    assert "--plot-column" not in help_text
    assert "locked_test_identity_source" not in _load_config()["paths"]

    missing_source = _run_cli(
        "freeze-test-identity", "--config", str(CONFIG_PATH)
    )
    assert missing_source.returncode != 0
    assert "--source" in missing_source.stderr
    assert "required" in missing_source.stderr.lower()


def test_frozen_config_uses_selected_fold_contained_vision_only_recipe():
    config = _load_config()
    encoder = config["experts"]["image_encoder"]
    recipe = encoder["fold_contained_adaptation"]["adaptation_recipe"]

    assert config["protocol_status"] == "frozen"
    assert encoder["strategy"] == "fold_contained_adaptation"
    assert recipe["vision_tower"]["unlocked_groups_from_end"] == 11
    assert recipe["text_tower"]["trainable"] is False
    assert len(recipe["prompts"]["values"]) == 18
    assert recipe["training"]["epochs"] == 5
    assert recipe["preprocessing"]["pixel_decoding"] == {
        "recipe_id": "legacy_opencv_bgr_439_v1",
        "decoder": "cv2.imread_color",
        "decoded_channel_order": "BGR",
        "pre_resize_size": [439, 439],
        "pre_resize_aspect_policy": "force_square",
        "pre_resize_interpolation": "cv2_inter_linear",
        "pil_conversion": "image_fromarray_without_channel_swap",
        "model_interpretation": "bgr_values_are_presented_as_rgb_channels",
        "deliberate_legacy_compatibility": True,
        "prohibit_silent_rgb_correction": True,
    }
    graph = config["experts"]["execution_graph"]
    assert graph["encoder_fits_per_training_seed"] == 6
    assert graph["expert_head_fits_per_training_seed"] == 18
    assert graph["total_encoder_fits_across_all_training_seeds"] == 24
    assert graph["total_expert_head_fits_across_all_training_seeds"] == 72
    assert len(graph["stages"]) == 6


def test_identity_sealing_projects_only_identities_and_is_create_once(tmp_path):
    config = _load_config()
    registry = tmp_path / "registry"
    config["paths"]["locked_test_registry_root"] = str(registry)
    config["locked_test_identity"]["expected_rows"] = 2
    config["locked_test_identity"]["expected_plots"] = 2
    config_path = _write_config(tmp_path / "protocol.yaml", config)
    source = tmp_path / "command_local_test_source.csv"
    pd.DataFrame(
        {
            "file": ["test/A.jpg", "test/B.jpg"],
            "plot_idx": ["T1", "T2"],
            "label_id": [3, 4],
            "A00": [0.25, 0.75],
        }
    ).to_csv(source, index=False)

    first = _run_cli(
        "freeze-test-identity",
        "--config",
        str(config_path),
        "--source",
        str(source),
    )
    assert first.returncode == 0, first.stdout + first.stderr
    active = json.loads((registry / "active_snapshot.json").read_text(encoding="utf-8"))
    manifest_path = registry / active["manifest_relative_path"]
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(source) not in manifest_text
    assert "label_id" not in manifest_text
    assert "A00" not in manifest_text

    second = _run_cli(
        "freeze-test-identity",
        "--config",
        str(config_path),
        "--source",
        str(source),
    )
    assert second.returncode != 0
    assert "already sealed" in (second.stdout + second.stderr)


def test_development_config_rejects_an_underlying_test_source_path(tmp_path):
    config = _load_config()
    config["paths"]["locked_test_identity_source"] = "/should/not/be/resolvable.parquet"
    config_path = _write_config(tmp_path / "unsafe_protocol.yaml", config)

    result = _run_cli("freeze-protocol", "--config", str(config_path))

    assert result.returncode != 0
    assert "must not contain an underlying locked-test path" in (
        result.stdout + result.stderr
    )


def test_development_cli_does_not_expose_underlying_test_table_arguments():
    for command in ("freeze-protocol", "validate-protocol"):
        result = _run_cli(command, "--help")
        assert result.returncode == 0
        help_text = result.stdout + result.stderr
        assert "--test-table" not in help_text
        assert "--test-source" not in help_text
        assert "--test-label" not in help_text


def test_cli_freezes_validates_and_detects_mutated_protocol_artifacts(tmp_path):
    config_path, test_source, protocol_root = _write_cli_protocol_fixture(tmp_path)

    seal = _run_cli(
        "freeze-test-identity",
        "--config",
        str(config_path),
        "--source",
        str(test_source),
    )
    assert seal.returncode == 0, seal.stdout + seal.stderr
    sealed = json.loads(seal.stdout)
    identity_manifest = Path(sealed["identity_manifest"])
    manifest_text = identity_manifest.read_text(encoding="utf-8")
    assert str(test_source) not in manifest_text
    assert "label_id" not in manifest_text
    assert "A00" not in manifest_text

    reseal = _run_cli(
        "freeze-test-identity",
        "--config",
        str(config_path),
        "--source",
        str(test_source),
    )
    assert reseal.returncode != 0
    assert "already sealed" in (reseal.stdout + reseal.stderr).lower()

    # Ordinary development freezing must remain functional after the underlying
    # test table disappears; it can read only the sealed identity manifest.
    test_source.unlink()

    freeze = _run_cli("freeze-protocol", "--config", str(config_path))
    assert freeze.returncode == 0, freeze.stdout + freeze.stderr
    frozen = json.loads(freeze.stdout)
    assert frozen["validation"]["validation_plot_count"] == 17
    assignments = pd.read_parquet(protocol_root / "development_assignments.parquet")
    train_assignments = assignments[assignments["development_role"] == "train"]
    assert set(train_assignments["train_oof_fold"].astype(int)) == {0, 1, 2, 3}
    validate = _run_cli(
        "validate-protocol",
        "--protocol-dir",
        str(protocol_root),
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
    validated = json.loads(validate.stdout)
    assert validated["status"] == "valid"
    assert validated["assignment_content_sha256"] == frozen["assignment_content_sha256"]

    second_freeze = _run_cli("freeze-protocol", "--config", str(config_path))
    assert second_freeze.returncode != 0
    assert "overwrite immutable" in (second_freeze.stdout + second_freeze.stderr).lower()

    balance_path = protocol_root / "split_balance.csv"
    os.chmod(balance_path, 0o644)
    balance_path.write_text(
        balance_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )
    tampered = _run_cli(
        "validate-protocol",
        "--protocol-dir",
        str(protocol_root),
    )
    assert tampered.returncode != 0
    assert "hash mismatch" in (tampered.stdout + tampered.stderr).lower()


def test_locked_predict_and_score_synthetic_lifecycle_is_immutable(tmp_path):
    identity_source = _identity_frame("locked", ["T1", "T2", "T3"])
    identity = protocol.build_identity_projection(
        identity_source,
        dataset_id="synthetic_locked_cs",
    )
    identity_manifest = protocol.build_locked_test_identity_manifest(
        identity_source,
        dataset_id="synthetic_locked_cs",
    )
    identity_path = tmp_path / "identity_manifest.json"
    identity_path.write_text(
        json.dumps(identity_manifest, sort_keys=True),
        encoding="utf-8",
    )

    bundle = {
        "schema_version": "synthetic_composite_bundle_v1",
        "artifact_role": "composite_inference_bundle",
        "protocol_id": "synthetic_protocol_v1",
        "identity_projection_sha256": identity_manifest["identity_projection_sha256"],
        "ontology_sha256": protocol.canonical_sha256(EXPECTED_ONTOLOGY),
        "final_expert_bundle_sha256": protocol.canonical_sha256(
            {"experts": "synthetic-fixed"}
        ),
        "selection_receipt_sha256": protocol.canonical_sha256(
            {"selection": "synthetic-sealed"}
        ),
        "frozen_router_bundle_sha256": protocol.canonical_sha256(
            {"router": "immutable", "threshold": "0x1.0p-2"}
        ),
        "threshold_specification": {
            "kind": "finite",
            "value_hex": "0x1.0000000000000p-2",
            "comparison": "strict_gt",
        },
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_bytes = json.dumps(bundle, sort_keys=True).encode("utf-8")
    bundle_path.write_bytes(bundle_bytes)

    features = identity[["row_uid"]].copy()
    features["training_seed"] = [1, 1, 1]
    features["raw_pred"] = [0, 1, 2]
    features["geo_pred"] = [1, 1, 3]
    features["q_rescue"] = [0.6, 0.7, 0.2]
    features["q_harm"] = [0.1, 0.1, 0.1]
    features["q_both_correct"] = [0.2, 0.1, 0.3]
    features["q_both_wrong"] = [0.1, 0.1, 0.4]
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)
    predictions_path = tmp_path / "predictions.parquet"

    predict = _run_cli(
        "locked-predict",
        "--synthetic-fixture",
        "--bundle",
        str(bundle_path),
        "--identity-manifest",
        str(identity_path),
        "--features",
        str(features_path),
        "--output",
        str(predictions_path),
    )
    assert predict.returncode == 0, predict.stdout + predict.stderr
    assert bundle_path.read_bytes() == bundle_bytes
    predictions = pd.read_parquet(predictions_path)
    assert predictions["final_pred"].tolist() == [1, 1, 2]

    labels = identity[["row_uid"]].copy()
    labels["label_id_dense"] = [1, 0, 2]
    labels_path = tmp_path / "labels.parquet"
    labels.to_parquet(labels_path, index=False)
    receipt_path = tmp_path / "score_receipt.json"
    score = _run_cli(
        "locked-score",
        "--synthetic-fixture",
        "--predictions",
        str(predictions_path),
        "--labels",
        str(labels_path),
        "--output",
        str(receipt_path),
    )
    assert score.returncode == 0, score.stdout + score.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["metrics"]["top1_accuracy"] == pytest.approx(2 / 3)

    second_score = _run_cli(
        "locked-score",
        "--synthetic-fixture",
        "--predictions",
        str(predictions_path),
        "--labels",
        str(labels_path),
        "--output",
        str(receipt_path),
    )
    assert second_score.returncode != 0
    assert "overwrite immutable" in (second_score.stdout + second_score.stderr).lower()


@pytest.mark.parametrize(
    "command,forbidden_flag",
    [
        ("locked-predict", "--labels"),
        ("locked-predict", "--fit"),
        ("locked-score", "--fit"),
        ("locked-score", "--recalibrate"),
        ("locked-score", "--threshold"),
        ("locked-score", "--policy"),
        ("locked-score", "--scientific-override"),
    ],
)
def test_locked_commands_do_not_expose_forbidden_capabilities(
    command: str,
    forbidden_flag: str,
):
    result = _run_cli(command, "--help")
    assert result.returncode == 0
    assert forbidden_flag not in (result.stdout + result.stderr)


def test_runner_can_be_imported_without_fitting_dependencies():
    spec = importlib.util.spec_from_file_location("geo_helpfulness_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert not hasattr(runner, "train_and_evaluate")
    assert not hasattr(runner, "fit_router")
