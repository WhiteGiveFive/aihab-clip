"""Full-shape synthetic coverage for the frozen RF policy application adapter."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_relaxed_rf as stage
from multimodal.geo_helpfulness_legacy_capacity_report import POLICIES as OLD_POLICIES
from test_multimodal_geo_relaxed_guardrails_predictions import (
    ACTION_KEYS,
    ACTION_SCHEMA,
    COMMON_FIELDS,
    EXPERT_SCHEMA,
    KEYS,
    specification,
)


ROOT = Path("/synthetic-relaxed-rf-output")
SOURCE = Path("/synthetic-capacity-legacy-input")
ROUTER_SCHEMA = pa.schema(
    [
        *COMMON_FIELDS,
        pa.field("policy_id", pa.string(), nullable=False),
        pa.field("learner", pa.string(), nullable=False),
        pa.field("variant_id", pa.string(), nullable=False),
        pa.field("config_id", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ]
)


@pytest.fixture(scope="module")
def synthetic_inputs():
    image = np.tile(np.repeat(np.arange(1347), 4), 2)
    seed = np.tile(np.arange(1, 5, dtype=np.int8), 2694)
    family = np.repeat(np.arange(2), 1347 * 4)
    raw = ((image + seed + family) % 18).astype(np.int8)
    geo = np.where(image % 5 == 0, raw, (raw + 1) % 18).astype(np.int8)
    experts = pd.DataFrame(
        {
            "schema_version": "synthetic.capacity_legacy.experts.v1",
            "study_id": "synthetic_frozen_capacity_legacy",
            "checkpoint_family": np.asarray(stage.legacy.FAMILIES)[family],
            "row_uid": [f"image-{index:04d}" for index in image],
            "plot_idx": [f"plot-{index % 531:03d}" for index in image],
            "training_seed": seed,
            "raw_pred": raw,
            "geo_pred": geo,
        }
    )
    score_values = np.array([-0.25, 0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    common = [field.name for field in COMMON_FIELDS]
    routers, policies = [], {}
    for learner_index, learner in enumerate(("lr", "rf", "xgb", "mlp")):
        for variant_index, variant in enumerate(stage.VARIANTS):
            if learner == "lr":
                source_id = variant
            elif learner == "mlp":
                source_id = f"mlp__{variant}"
            else:
                source_id = f"capacity__{learner}__{variant}"
            config = f"{learner}__{variant}__synthetic_frozen"
            router = experts[common].copy()
            router["schema_version"] = "synthetic.capacity_legacy.routers.v1"
            router["policy_id"], router["learner"] = source_id, learner
            router["variant_id"], router["config_id"] = variant, config
            router["score"] = score_values[
                (image + 2 * seed + 3 * family + variant_index + learner_index) % 5
            ]
            routers.append(router)
            if learner != "rf":
                continue
            for minimum, base in ((20, 0.0), (10, 0.25), (0, 0.5)):
                for numerator in (5, 7, 9):
                    guardrail = f"plots_{minimum}__harm_0p{numerator}"
                    name = f"relaxed__rf__{variant}__{guardrail}"
                    threshold = (
                        None
                        if numerator == 9
                        else base + (0.25 if numerator == 7 else 0.0)
                    )
                    policies[name] = {
                        "policy_id": name,
                        "learner": "rf",
                        "original_policy_id": source_id,
                        "variant_id": variant,
                        "config_id": config,
                        "guardrail_id": guardrail,
                        "minimum_plots": minimum,
                        "harm_numerator": numerator,
                        "harm_denominator": 10,
                        "minimum_coverage_fraction": 0.01,
                        "threshold_specification": specification(threshold),
                    }
    old_actions = []
    for index, name in enumerate(OLD_POLICIES):
        frame = experts[common].copy()
        frame["schema_version"] = "synthetic.capacity_legacy.actions.v1"
        frame["policy_id"] = name
        frame["score"] = score_values[(image + seed + family + index) % 5]
        spec = specification(None if name == "never_intervene" else 0.25)
        frame["threshold_specification"] = stage.canonical_bytes(spec).decode()
        frame["acted"] = (
            np.zeros(len(frame), dtype=bool)
            if spec["kind"] == "never_intervene"
            else (frame.score.to_numpy() > 0.25) & (raw != geo)
        )
        frame["raw_pred"], frame["geo_pred"] = raw, geo
        frame["final_pred"] = np.where(frame.acted, geo, raw)
        old_actions.append(frame)
    frames = {
        "experts": experts.sample(frac=1, random_state=321).reset_index(drop=True),
        "routers": pd.concat(routers, ignore_index=True)
        .sample(frac=1, random_state=437)
        .reset_index(drop=True),
        "actions": pd.concat(old_actions, ignore_index=True)
        .sort_values(ACTION_KEYS, kind="stable")
        .reset_index(drop=True),
    }
    tables = {
        name: pa.Table.from_pandas(
            frames[name], schema=schema, preserve_index=False
        ).replace_schema_metadata({b"synthetic_source": name.encode()})
        for name, schema in (
            ("experts", EXPERT_SCHEMA),
            ("routers", ROUTER_SCHEMA),
            ("actions", ACTION_SCHEMA),
        )
    }
    return {
        "frames": frames,
        "tables": tables,
        "policies": policies,
        "old_declaration": {"declaration_sha256": "synthetic-old-declaration"},
        "selection_manifest": {"manifest_sha256": "synthetic-rf-selection"},
        "declaration": {
            "paths": {"legacy_root": str(SOURCE)},
            "legacy_declaration_sha256": "synthetic-old-declaration",
            "legacy_prediction_manifest_sha256": "synthetic-old-predictions",
            "legacy_score_manifest_sha256": "synthetic-old-scores",
        },
    }


def wire_adapter(monkeypatch, inputs, frames=None):
    reads, metadata_calls = [], []

    def load_policies(root, declaration):
        assert root == ROOT
        assert declaration == inputs["declaration"]
        return inputs["selection_manifest"], inputs["policies"]

    def load_declaration(source):
        assert source == SOURCE
        metadata_calls.append("declaration")
        return inputs["old_declaration"]

    def manifest(root, declaration, kind, expected):
        assert root == SOURCE / kind
        assert declaration == inputs["old_declaration"]
        assert kind in {"predictions", "scores"}
        assert expected == (
            stage.predecessor.PREDICTION_FILES
            if kind == "predictions"
            else stage.predecessor.SCORE_FILES
        )
        metadata_calls.append(kind)
        return {"manifest_sha256": f"synthetic-old-{kind}"}

    def predictions(source):
        assert source == SOURCE
        metadata_calls.append("validated_predictions")
        return None, None, inputs["frames"] if frames is None else frames

    def read_table(path):
        path = Path(path)
        assert path in {
            SOURCE / "predictions/experts.parquet",
            SOURCE / "predictions/routers.parquet",
        }
        reads.append(path)
        return inputs["tables"][path.stem]

    def read_schema(path):
        assert Path(path) == SOURCE / "predictions/actions.parquet"
        reads.append(Path(path))
        return ACTION_SCHEMA

    monkeypatch.setattr(stage, "_load_policies", load_policies)
    monkeypatch.setattr(stage.predecessor, "load_declaration", load_declaration)
    monkeypatch.setattr(stage.predecessor, "_manifest", manifest)
    monkeypatch.setattr(stage.predecessor, "_validate_predictions", predictions)
    monkeypatch.setattr(stage.pq, "read_table", read_table)
    monkeypatch.setattr(stage.pq, "read_schema", read_schema)
    return reads, metadata_calls


def test_actual_rf_adapter_preserves_sources_and_applies_every_frozen_policy(
    synthetic_inputs, monkeypatch
):
    inputs = synthetic_inputs
    before = {name: frame.copy(deep=True) for name, frame in inputs["frames"].items()}
    before_policies = deepcopy(inputs["policies"])
    reads, metadata_calls = wire_adapter(monkeypatch, inputs)
    manifest, policies, tables = stage._test_tables(ROOT, inputs["declaration"])
    assert manifest == inputs["selection_manifest"]
    assert policies == before_policies
    assert {name: table.num_rows for name, table in tables.items()} == {
        "experts": 10776,
        "routers": 129312,
        "actions": 474144,
    }
    assert before["routers"].groupby("learner").size().to_dict() == {
        learner: 32328 for learner in ("lr", "rf", "xgb", "mlp")
    }
    for name in ("experts", "routers"):
        assert tables[name].equals(inputs["tables"][name], check_metadata=True)
    actions = tables["actions"].to_pandas()
    old = actions.loc[actions.policy_id.isin(OLD_POLICIES)].reset_index(drop=True)
    assert len(old) == 183192
    pd.testing.assert_frame_equal(old, before["actions"])
    new = actions.loc[actions.policy_id.isin(policies)]
    assert len(new) == 290952
    counts = new.groupby(["policy_id", "checkpoint_family", "training_seed"]).size()
    assert len(counts) == 27 * 2 * 4 and counts.eq(1347).all()
    assert new.schema_version.eq(stage.SCHEMA + ".actions").all()
    assert new.study_id.eq(stage.STUDY_ID).all()
    expert_groups = {
        family: frame.sort_values(KEYS).reset_index(drop=True)
        for family, frame in before["experts"].groupby("checkpoint_family")
    }
    router_groups = {
        identity: frame.sort_values(KEYS).reset_index(drop=True)
        for identity, frame in before["routers"].groupby(
            ["checkpoint_family", "policy_id"]
        )
    }
    equality_count = agreement_above_threshold = never_count = 0
    for (family, name), frame in new.groupby(["checkpoint_family", "policy_id"]):
        frame = frame.reset_index(drop=True)
        policy = policies[name]
        expert = expert_groups[family]
        router = router_groups[family, policy["original_policy_id"]]
        spec = policy["threshold_specification"]
        assert router.learner.eq("rf").all()
        assert router.config_id.eq(policy["config_id"]).all()
        assert router.variant_id.eq(policy["variant_id"]).all()
        np.testing.assert_array_equal(frame.score, router.score)
        np.testing.assert_array_equal(frame.raw_pred, expert.raw_pred)
        np.testing.assert_array_equal(frame.geo_pred, expert.geo_pred)
        pd.testing.assert_frame_equal(
            frame[KEYS + ["plot_idx"]], expert[KEYS + ["plot_idx"]]
        )
        assert frame.threshold_specification.eq(
            stage.canonical_bytes(spec).decode()
        ).all()
        if spec["kind"] == "never_intervene":
            expected = np.zeros(len(frame), dtype=bool)
            never_count += 1
        else:
            threshold = float.fromhex(spec["value_hex"])
            expected = (router.score.to_numpy() > threshold) & (
                expert.raw_pred.to_numpy() != expert.geo_pred.to_numpy()
            )
            equal = frame.score.eq(threshold)
            agreement = frame.raw_pred.eq(frame.geo_pred) & frame.score.gt(threshold)
            equality_count += int(equal.sum())
            agreement_above_threshold += int(agreement.sum())
            assert not frame.loc[equal | agreement, "acted"].any()
        np.testing.assert_array_equal(frame.acted, expected)
        np.testing.assert_array_equal(
            frame.final_pred,
            np.where(expected, expert.geo_pred.to_numpy(), expert.raw_pred.to_numpy()),
        )
    assert equality_count > 0 and agreement_above_threshold > 0
    assert never_count == 3 * 3 * 2
    for name, frame in before.items():
        pd.testing.assert_frame_equal(inputs["frames"][name], frame)
    assert inputs["policies"] == before_policies
    assert set(reads) == {
        SOURCE / f"predictions/{name}.parquet"
        for name in ("experts", "routers", "actions")
    }
    assert metadata_calls == [
        "declaration",
        "predictions",
        "scores",
        "validated_predictions",
    ]


@pytest.mark.parametrize(
    "damage", ["config_id", "learner", "row_uid", "plot_idx", "missing_row"]
)
def test_actual_rf_adapter_rejects_wrong_config_learner_or_identity(
    synthetic_inputs, monkeypatch, damage
):
    frames = dict(synthetic_inputs["frames"])
    routers = frames["routers"].copy(deep=True)
    mask = routers.checkpoint_family.eq(
        stage.legacy.FAMILIES[0]
    ) & routers.policy_id.eq(f"capacity__rf__{stage.VARIANTS[0]}")
    index = routers.index[mask][0]
    if damage == "missing_row":
        routers = routers.drop(index=index)
    else:
        routers.loc[index, damage] = (
            "lr" if damage == "learner" else "wrong-synthetic-value"
        )
    frames["routers"] = routers
    reads, metadata_calls = wire_adapter(monkeypatch, synthetic_inputs, frames)
    with pytest.raises(
        ValueError, match="RF router identity or selected configuration differs"
    ):
        stage._test_tables(ROOT, synthetic_inputs["declaration"])
    assert reads == []
    assert metadata_calls == [
        "declaration",
        "predictions",
        "scores",
        "validated_predictions",
    ]
