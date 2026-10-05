"""Full-shape synthetic tests of the frozen-policy application adapter."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from multimodal import geo_helpfulness_relaxed_guardrails as stage
from multimodal.geo_helpfulness_router_m4_common import POLICIES as OLD_POLICIES


ROOT = Path("/synthetic-relaxed-output")
SOURCE = Path("/synthetic-legacy-input")
KEYS = ["row_uid", "training_seed"]
ACTION_KEYS = ["checkpoint_family", "policy_id", *KEYS]
COMMON_FIELDS = [
    pa.field("schema_version", pa.string(), nullable=False),
    pa.field("study_id", pa.string(), nullable=False),
    pa.field("checkpoint_family", pa.string(), nullable=False),
    pa.field("row_uid", pa.string(), nullable=False),
    pa.field("plot_idx", pa.string(), nullable=False),
    pa.field("training_seed", pa.int8(), nullable=False),
]
EXPERT_SCHEMA = pa.schema(
    [
        *COMMON_FIELDS,
        pa.field("raw_pred", pa.int8(), nullable=False),
        pa.field("geo_pred", pa.int8(), nullable=False),
    ]
)
ROUTER_SCHEMA = pa.schema(
    [
        *COMMON_FIELDS,
        pa.field("variant_id", pa.string(), nullable=False),
        pa.field("C_hex", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ]
)
ACTION_SCHEMA = pa.schema(
    [
        *COMMON_FIELDS,
        pa.field("policy_id", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("threshold_specification", pa.string(), nullable=False),
        pa.field("acted", pa.bool_(), nullable=False),
        pa.field("raw_pred", pa.int8(), nullable=False),
        pa.field("geo_pred", pa.int8(), nullable=False),
        pa.field("final_pred", pa.int8(), nullable=False),
    ]
)


def specification(value):
    return {
        "kind": "never_intervene" if value is None else "finite",
        "comparison": "strict_gt",
        "value_hex": None if value is None else float(value).hex(),
    }


@pytest.fixture(scope="module")
def synthetic_inputs():
    image = np.tile(np.repeat(np.arange(1347), 4), 2)
    seed = np.tile(np.arange(1, 5, dtype=np.int8), 2694)
    family = np.repeat(np.arange(2), 1347 * 4)
    raw = ((image + seed + family) % 18).astype(np.int8)
    geo = np.where(image % 5 == 0, raw, (raw + 1) % 18).astype(np.int8)
    experts = pd.DataFrame(
        {
            "schema_version": "synthetic.legacy.experts.v1",
            "study_id": "synthetic_frozen_legacy",
            "checkpoint_family": np.asarray(stage.legacy.FAMILIES)[family],
            "row_uid": [f"image-{index:04d}" for index in image],
            "plot_idx": [f"plot-{index % 531:03d}" for index in image],
            "training_seed": seed,
            "raw_pred": raw,
            "geo_pred": geo,
        }
    )
    routers, policies = [], {}
    score_values = np.array([-0.25, 0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    common_columns = [field.name for field in COMMON_FIELDS]
    for index, variant in enumerate(stage.VARIANTS):
        C = (0.01, 0.1, 1.0)[index]
        router = experts[common_columns].copy()
        router["schema_version"] = "synthetic.legacy.routers.v1"
        router["variant_id"], router["C_hex"] = variant, C.hex()
        router["score"] = score_values[(image + 2 * seed + 3 * family + index) % 5]
        routers.append(router)
        for minimum, base in ((20, 0.0), (10, 0.25), (0, 0.5)):
            for numerator in (5, 7, 9):
                guardrail = f"plots_{minimum}__harm_0p{numerator}"
                name = f"relaxed__{variant}__{guardrail}"
                threshold = (
                    None if numerator == 9 else base + (0.25 if numerator == 7 else 0)
                )
                policies[name] = {
                    "policy_id": name,
                    "variant_id": variant,
                    "guardrail_id": guardrail,
                    "minimum_plots": minimum,
                    "harm_numerator": numerator,
                    "harm_denominator": 10,
                    "minimum_coverage_fraction": 0.01,
                    "selected_C": C,
                    "selected_C_hex": C.hex(),
                    "threshold_specification": specification(threshold),
                }
    old_actions = []
    for index, name in enumerate(OLD_POLICIES):
        frame = experts[common_columns].copy()
        frame["schema_version"] = "synthetic.legacy.actions.v1"
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
        "experts": experts.sample(frac=1, random_state=29).reset_index(drop=True),
        "routers": pd.concat(routers, ignore_index=True)
        .sample(frac=1, random_state=43)
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
        "selection_manifest": {"manifest_sha256": "synthetic-selection"},
        "declaration": {
            "paths": {"legacy_root": str(SOURCE)},
            "legacy_declaration_sha256": "synthetic-old-declaration",
            "legacy_prediction_manifest_sha256": "synthetic-old-predictions",
            "legacy_score_manifest_sha256": "synthetic-old-scores",
        },
    }


def wire_adapter(monkeypatch, inputs, frames=None):
    reads = []

    def load_policies(root, declaration):
        assert root == ROOT
        assert declaration == inputs["declaration"]
        return inputs["selection_manifest"], inputs["policies"]

    def metadata(source):
        assert source == SOURCE
        return (
            {"declaration_sha256": "synthetic-old-declaration"},
            {"manifest_sha256": "synthetic-old-predictions"},
            {"manifest_sha256": "synthetic-old-scores"},
        )

    def predictions(source):
        assert source == SOURCE
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
    monkeypatch.setattr(stage.predecessor, "_legacy_metadata", metadata)
    monkeypatch.setattr(stage.legacy, "validate_predictions", predictions)
    monkeypatch.setattr(stage.pq, "read_table", read_table)
    monkeypatch.setattr(stage.pq, "read_schema", read_schema)
    return reads


def test_actual_adapter_applies_all_policies_and_preserves_sources(
    synthetic_inputs, monkeypatch
):
    inputs = synthetic_inputs
    before = {name: frame.copy(deep=True) for name, frame in inputs["frames"].items()}
    before_policies = deepcopy(inputs["policies"])
    reads = wire_adapter(monkeypatch, inputs)
    manifest, policies, tables = stage._test_tables(ROOT, inputs["declaration"])
    assert manifest == inputs["selection_manifest"]
    assert policies == before_policies
    assert {name: table.num_rows for name, table in tables.items()} == {
        "experts": 10776,
        "routers": 32328,
        "actions": 377160,
    }
    for name in ("experts", "routers"):
        assert tables[name].equals(inputs["tables"][name], check_metadata=True)
    actions = tables["actions"].to_pandas()
    old = actions.loc[actions.policy_id.isin(OLD_POLICIES)].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, before["actions"])
    new = actions.loc[actions.policy_id.isin(policies)]
    counts = new.groupby(["policy_id", "checkpoint_family", "training_seed"]).size()
    assert len(counts) == 27 * 2 * 4
    assert counts.eq(1347).all()
    assert new.schema_version.eq(stage.SCHEMA + ".actions").all()
    assert new.study_id.eq(stage.STUDY_ID).all()
    expert_groups = {
        family: frame.sort_values(KEYS).reset_index(drop=True)
        for family, frame in before["experts"].groupby("checkpoint_family")
    }
    router_groups = {
        identity: frame.sort_values(KEYS).reset_index(drop=True)
        for identity, frame in before["routers"].groupby(
            ["checkpoint_family", "variant_id"]
        )
    }
    equality_count = agreement_above_threshold = 0
    for (family, name), frame in new.groupby(["checkpoint_family", "policy_id"]):
        frame = frame.reset_index(drop=True)
        policy = policies[name]
        expert = expert_groups[family]
        router = router_groups[family, policy["variant_id"]]
        spec = policy["threshold_specification"]
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
    assert equality_count > 0
    assert agreement_above_threshold > 0
    for name, frame in before.items():
        pd.testing.assert_frame_equal(inputs["frames"][name], frame)
    assert inputs["policies"] == before_policies
    assert set(reads) == {
        SOURCE / f"predictions/{name}.parquet"
        for name in ("experts", "routers", "actions")
    }


@pytest.mark.parametrize("damage", ["selected_C", "row_uid", "plot_idx", "missing_row"])
def test_actual_adapter_rejects_router_c_or_identity_mismatch(
    synthetic_inputs, monkeypatch, damage
):
    frames = dict(synthetic_inputs["frames"])
    routers = frames["routers"].copy(deep=True)
    mask = routers.checkpoint_family.eq(
        stage.legacy.FAMILIES[0]
    ) & routers.variant_id.eq(stage.VARIANTS[0])
    index = routers.index[mask][0]
    if damage == "selected_C":
        routers.loc[index, "C_hex"] = (10.0).hex()
    elif damage == "missing_row":
        routers = routers.drop(index=index)
    else:
        routers.loc[index, damage] = "wrong-synthetic-identity"
    frames["routers"] = routers
    wire_adapter(monkeypatch, synthetic_inputs, frames)
    with pytest.raises(ValueError, match="router identity or selected C differs"):
        stage._test_tables(ROOT, synthetic_inputs["declaration"])
