"""Apply-only, label-blind JSON state for the M4 router candidates.

This module deliberately imports no fitting or parent-artifact modules. A later
prediction process can validate and apply a model without opening training data
or loading scikit-learn, SciPy, Torch, or a pickle.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "geo_helpfulness.router_candidate.v2"
LEGACY_SCHEMA_VERSION = "geo_helpfulness.router_candidate.v1"
SCHEMA_MAX_ITER = {LEGACY_SCHEMA_VERSION: 1000, SCHEMA_VERSION: 5000}
TARGET_ORDER = ("rescue", "harm", "both_correct", "both_wrong")
C_GRID = (0.01, 0.1, 1.0, 10.0)
TRAINING_SEEDS = (1, 2, 3, 4)
VARIANTS = {
    "primary_full727": 727,
    "ablation_no_pairs79": 79,
    "ablation_numeric25": 25,
}
C_TOKENS = {0.01: "0p01", 0.1: "0p1", 1.0: "1", 10.0: "10"}
NUMERIC_INPUT_COLUMNS = (
    "image_geo_agree",
    "image_raw_agree",
    "geo_raw_agree",
    "image_geo_top3_overlap",
    "raw_rank_of_geo_pred",
    "image_confidence",
    "geo_confidence",
    "raw_confidence",
    "image_entropy",
    "geo_entropy",
    "raw_entropy",
    "image_top2_margin",
    "geo_top2_margin",
    "raw_top2_margin",
    "geo_minus_image_confidence",
    "geo_minus_raw_confidence",
    "geo_minus_image_entropy",
    "geo_minus_raw_entropy",
    "geo_minus_image_margin",
    "geo_minus_raw_margin",
    "image_geo_jsd",
    "image_geo_total_variation",
    "image_probability_at_geo_pred",
    "geo_probability_at_image_pred",
    "raw_probability_at_geo_pred",
)
CATEGORICAL_FEATURE_SIZES = (
    ("image_pred", 18),
    ("geo_pred", 18),
    ("raw_pred", 18),
    ("image_geo_pred_pair", 324),
    ("geo_raw_pred_pair", 324),
)
FROZEN_FIT_PARAMETERS = {
    "penalty": "l2",
    "solver": "lbfgs",
    "multi_class": "multinomial",
    "fit_intercept": True,
    "class_weight": None,
    "dual": False,
    "warm_start": False,
    "n_jobs": 1,
    "max_iter": 5000,
    "tol": 1.0e-8,
    "random_state": None,
}
IMPLEMENTATION = {
    "library": "sklearn",
    "version": "1.5.2",
    "compute_dtype": "float64",
    "threadpool_limit": 1,
}
STATE_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "training_seed",
        "variant_id",
        "C",
        "C_hex",
        "target_order",
        "classes",
        "fit_role",
        "fit_row_count",
        "fit_row_identity_sha256",
        "study_spec_sha256",
        "dataset_manifest_sha256",
        "feature_input_names",
        "feature_indices",
        "feature_names",
        "fit_parameters",
        "implementation",
        "coef",
        "intercept",
        "n_iter",
        "state_sha256",
    }
)
PARENT_KEYS = frozenset(
    {
        "protocol_id",
        "fit_row_identity_sha256",
        "study_spec_sha256",
        "dataset_manifest_sha256",
    }
)


class CandidateStateError(ValueError):
    """A candidate state or its inference input violates the frozen contract."""


def full_feature_columns() -> list[str]:
    """Return the exact sealed-dataset input order, without parent imports."""
    return [f"scaled__{name}" for name in NUMERIC_INPUT_COLUMNS] + [
        f"onehot__{name}__{category}"
        for name, size in CATEGORICAL_FEATURE_SIZES
        for category in range(size)
    ]


def variant_feature_indices(variant_id: str) -> list[int]:
    """Select frozen column blocks, without re-fitting any preprocessing."""
    if not isinstance(variant_id, str) or variant_id not in VARIANTS:
        raise CandidateStateError(f"unknown router variant: {variant_id!r}")
    return list(range(VARIANTS[variant_id]))


def candidate_filename(variant_id: str, seed: int, C: float) -> str:
    """Return one path-traversal-safe name in the fixed 48-candidate grid."""
    variant_feature_indices(variant_id)
    if type(seed) is not int or seed not in TRAINING_SEEDS:
        raise CandidateStateError("invalid candidate seed")
    if type(C) is not float or C not in C_GRID:
        raise CandidateStateError("invalid candidate C")
    return f"{variant_id}__seed_{seed}__C_{C_TOKENS[C]}.json"


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateStateError(
            "candidate state must contain finite JSON values"
        ) from exc


def candidate_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash all state fields except the self-hash, using canonical UTF-8 JSON."""
    if not isinstance(state, dict):
        raise CandidateStateError("candidate state must be a JSON object")
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    return canonical_sha256(payload)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value using the candidate artifact's canonical encoding."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def seal_candidate_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy, self-hash, and validate a complete state; never alter the caller."""
    result = deepcopy(state)
    result["state_sha256"] = candidate_state_sha256(result)
    validate_candidate_state(result)
    return result


def _is_integer(value: Any) -> bool:
    return type(value) is int


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CandidateStateError(f"{name} must be a lowercase SHA-256 digest")


def _numeric_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(value, list):
        raise CandidateStateError(f"{name} must be a JSON array")
    elements = value
    if len(shape) == 2:
        if len(value) != shape[0] or any(not isinstance(row, list) for row in value):
            raise CandidateStateError(f"{name} must have shape {shape}")
        if any(len(row) != shape[1] for row in value):
            raise CandidateStateError(f"{name} must have shape {shape}")
        elements = [element for row in value for element in row]
    if any(type(item) not in (int, float) for item in elements):
        raise CandidateStateError(f"{name} must contain JSON numbers, not booleans")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise CandidateStateError(f"invalid numeric array {name}") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise CandidateStateError(f"{name} must be finite with shape {shape}")
    return array


def validate_candidate_state(state: dict[str, Any]) -> None:
    """Validate a self-contained candidate without reading any dataset or label."""
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise CandidateStateError("candidate state has missing or unknown keys")
    if (
        not isinstance(state["schema_version"], str)
        or state["schema_version"] not in SCHEMA_MAX_ITER
    ):
        raise CandidateStateError("candidate state schema version mismatch")
    if not isinstance(state["protocol_id"], str) or not state["protocol_id"].strip():
        raise CandidateStateError("protocol_id must be a nonempty string")
    if (
        not _is_integer(state["training_seed"])
        or state["training_seed"] not in TRAINING_SEEDS
    ):
        raise CandidateStateError("training_seed must be one of 1, 2, 3, 4")
    indices = variant_feature_indices(state["variant_id"])
    if type(state["C"]) is not float or state["C"] not in C_GRID:
        raise CandidateStateError("C must be a float in the frozen candidate grid")
    if state["C_hex"] != state["C"].hex():
        raise CandidateStateError("C_hex differs from C")
    if state["target_order"] != list(TARGET_ORDER):
        raise CandidateStateError("four-state output order mismatch")
    if _canonical_json(state["classes"]) != "[0,1,2,3]":
        raise CandidateStateError("classes must be integer IDs in fixed target order")
    if state["fit_role"] != "development_train_oof":
        raise CandidateStateError("candidate was not fit on development_train_oof")
    if not _is_integer(state["fit_row_count"]) or state["fit_row_count"] < 4:
        raise CandidateStateError("fit_row_count must be an integer >= 4")
    for name in (
        "fit_row_identity_sha256",
        "study_spec_sha256",
        "dataset_manifest_sha256",
        "state_sha256",
    ):
        _digest(state[name], name)
    columns = full_feature_columns()
    if state["feature_input_names"] != columns:
        raise CandidateStateError("full 727-column input schema mismatch")
    if _canonical_json(state["feature_indices"]) != _canonical_json(indices):
        raise CandidateStateError("variant feature indices mismatch")
    if state["feature_names"] != [columns[index] for index in indices]:
        raise CandidateStateError("selected feature names mismatch")
    maximum_iterations = SCHEMA_MAX_ITER[state["schema_version"]]
    expected_parameters = dict(
        FROZEN_FIT_PARAMETERS, max_iter=maximum_iterations, C=state["C"]
    )
    if _canonical_json(state["fit_parameters"]) != _canonical_json(expected_parameters):
        raise CandidateStateError("frozen optimizer parameters mismatch")
    if _canonical_json(state["implementation"]) != _canonical_json(IMPLEMENTATION):
        raise CandidateStateError("candidate implementation identity mismatch")
    _numeric_array(state["coef"], (4, len(indices)), "coef")
    _numeric_array(state["intercept"], (4,), "intercept")
    iterations = state["n_iter"]
    if (
        not isinstance(iterations, list)
        or len(iterations) != 1
        or not _is_integer(iterations[0])
        or not 0 <= iterations[0] < maximum_iterations
    ):
        raise CandidateStateError(
            "n_iter must show convergence before the iteration cap"
        )
    if candidate_state_sha256(state) != state["state_sha256"]:
        raise CandidateStateError("candidate state self-hash mismatch")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateStateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CandidateStateError(f"nonfinite JSON constant: {value}")


def read_candidate_state(
    path: str | Path,
    *,
    expected_parents: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load one regular JSON file and optionally bind supplied parent identities.

    ``expected_parents`` contains expected hash/ID strings supplied by the caller;
    this function never opens a parent artifact to obtain them.
    """
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise CandidateStateError("candidate state must be a regular nonsymlink file")
    try:
        state = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateStateError("cannot read candidate JSON state") from exc
    validate_candidate_state(state)
    if expected_parents is not None:
        if not isinstance(expected_parents, Mapping) or not set(
            expected_parents
        ).issubset(PARENT_KEYS):
            raise CandidateStateError("unknown expected-parent identity key")
        for name, expected in expected_parents.items():
            if state[name] != expected:
                raise CandidateStateError(f"candidate parent identity mismatch: {name}")
    return state


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise CandidateStateError(f"expected regular nonsymlink file: {path.name}")
    try:
        data = path.read_bytes()
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateStateError(f"cannot read JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise CandidateStateError(f"JSON artifact is not an object: {path.name}")
    return value, data


def load_candidate_set_for_inference(
    root: str | Path,
    *,
    expected_dataset_manifest_sha256: str,
    expected_study_spec_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify and load all 48 models without accessing labels or training files.

    Only files inside the committed candidate directory are opened. The caller
    supplies the expected dataset identity from its own approved lineage record.
    """
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise CandidateStateError("candidate bundle must be a nonsymlink directory")
    _digest(expected_dataset_manifest_sha256, "expected_dataset_manifest_sha256")
    if expected_study_spec_sha256 is not None:
        _digest(expected_study_spec_sha256, "expected_study_spec_sha256")
    expected = {
        candidate_filename(variant, seed, C): (variant, seed, C)
        for variant in VARIANTS
        for seed in TRAINING_SEEDS
        for C in C_GRID
    }
    filenames = set(expected) | {
        "study_spec.json",
        "training_report.json",
        "manifest.json",
    }
    if {path.name for path in directory.iterdir()} != filenames:
        raise CandidateStateError("candidate bundle file allowlist mismatch")
    manifest, _ = _read_json_bytes(directory / "manifest.json")
    if manifest.get("schema_version") != "geo_helpfulness.router_candidate_manifest.v1":
        raise CandidateStateError("candidate manifest schema mismatch")
    for name in ("manifest_sha256", "study_spec_sha256", "dataset_manifest_sha256"):
        _digest(manifest.get(name), name)
    if manifest["manifest_sha256"] != canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise CandidateStateError("candidate manifest self-hash mismatch")
    if manifest["dataset_manifest_sha256"] != expected_dataset_manifest_sha256:
        raise CandidateStateError("candidate manifest dataset parent mismatch")
    if (
        expected_study_spec_sha256 is not None
        and manifest["study_spec_sha256"] != expected_study_spec_sha256
    ):
        raise CandidateStateError("candidate manifest study parent mismatch")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(expected):
        raise CandidateStateError(
            "manifest must contain the complete 48-candidate grid"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "study_spec.json",
        "training_report.json",
    }:
        raise CandidateStateError("candidate manifest ancillary artifact list mismatch")
    ancillary = {}
    for name, record in artifacts.items():
        if not isinstance(record, dict) or "file_sha256" not in record:
            raise CandidateStateError("missing ancillary artifact file hash")
        _digest(record["file_sha256"], f"{name}.file_sha256")
        value, data = _read_json_bytes(directory / name)
        if hashlib.sha256(data).hexdigest() != record["file_sha256"]:
            raise CandidateStateError(f"ancillary artifact hash mismatch: {name}")
        ancillary[name] = value
    study = ancillary["study_spec.json"]
    study_payload = {
        key: value for key, value in study.items() if key != "study_spec_sha256"
    }
    if (
        study.get("study_spec_sha256") != manifest["study_spec_sha256"]
        or canonical_sha256(study_payload) != manifest["study_spec_sha256"]
    ):
        raise CandidateStateError("study specification content hash mismatch")
    study_state_schemas = {
        "geo_helpfulness.router_candidate_study.v1": LEGACY_SCHEMA_VERSION,
        "geo_helpfulness.router_candidate_study.v2": SCHEMA_VERSION,
    }
    study_version = study.get("schema_version")
    if not isinstance(study_version, str) or study_version not in study_state_schemas:
        raise CandidateStateError("candidate study schema version mismatch")
    expected_state_schema = study_state_schemas[study_version]
    expected_parameters = dict(
        FROZEN_FIT_PARAMETERS, max_iter=SCHEMA_MAX_ITER[expected_state_schema]
    )
    if _canonical_json(study.get("model_parameters")) != _canonical_json(
        expected_parameters
    ):
        raise CandidateStateError("candidate study schema/optimizer budget mismatch")
    if (
        study.get("candidate_state_schema", expected_state_schema)
        != expected_state_schema
    ):
        raise CandidateStateError("candidate study/state schema mismatch")
    if "parent_model_parameters" in study and _canonical_json(
        study["parent_model_parameters"]
    ) != _canonical_json(
        dict(FROZEN_FIT_PARAMETERS, max_iter=SCHEMA_MAX_ITER[LEGACY_SCHEMA_VERSION])
    ):
        raise CandidateStateError(
            "candidate study amended an unsupported parent setting"
        )
    states = {}
    record_keys = {
        "file_sha256",
        "state_sha256",
        "variant_id",
        "training_seed",
        "C_hex",
    }
    for filename, (variant, seed, C) in expected.items():
        record = candidates[filename]
        if not isinstance(record, dict) or set(record) != record_keys:
            raise CandidateStateError(f"invalid candidate manifest entry: {filename}")
        _digest(record["file_sha256"], "candidate file_sha256")
        identity = {"variant_id": variant, "training_seed": seed, "C_hex": C.hex()}
        if _canonical_json({key: record[key] for key in identity}) != _canonical_json(
            identity
        ):
            raise CandidateStateError(
                f"candidate manifest identity mismatch: {filename}"
            )
        state, data = _read_json_bytes(directory / filename)
        if hashlib.sha256(data).hexdigest() != record["file_sha256"]:
            raise CandidateStateError(f"candidate file hash mismatch: {filename}")
        validate_candidate_state(state)
        if state["schema_version"] != expected_state_schema:
            raise CandidateStateError(
                "candidate state does not match the study optimizer budget"
            )
        if record["state_sha256"] != state["state_sha256"]:
            raise CandidateStateError(
                f"candidate manifest/state hash mismatch: {filename}"
            )
        if any(state[key] != value for key, value in identity.items()):
            raise CandidateStateError(f"candidate state identity mismatch: {filename}")
        if state["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"]:
            raise CandidateStateError(
                f"candidate state dataset parent mismatch: {filename}"
            )
        if state["study_spec_sha256"] != manifest["study_spec_sha256"]:
            raise CandidateStateError(
                f"candidate state study parent mismatch: {filename}"
            )
        states[filename] = state
    if len({state["protocol_id"] for state in states.values()}) != 1:
        raise CandidateStateError("candidate set contains mixed protocol identities")
    protocol_id = next(iter(states.values()))["protocol_id"]
    for payload in (manifest, study):
        if "protocol_id" in payload and payload["protocol_id"] != protocol_id:
            raise CandidateStateError(
                "candidate set protocol differs from parent metadata"
            )
    if (
        "dataset_manifest_sha256" in study
        and study["dataset_manifest_sha256"] != expected_dataset_manifest_sha256
    ):
        raise CandidateStateError("study specification dataset parent mismatch")
    for seed in TRAINING_SEEDS:
        seed_states = [
            state for state in states.values() if state["training_seed"] == seed
        ]
        if (
            len(
                {
                    (state["fit_row_count"], state["fit_row_identity_sha256"])
                    for state in seed_states
                }
            )
            != 1
        ):
            raise CandidateStateError(
                "candidate set has inconsistent fit rows within a seed"
            )
        if "seed_inputs" in study:
            source = study["seed_inputs"].get(str(seed), {})
            if (source.get("row_count"), source.get("row_identity_sha256")) != (
                seed_states[0]["fit_row_count"],
                seed_states[0]["fit_row_identity_sha256"],
            ):
                raise CandidateStateError(
                    "candidate fit rows differ from study metadata"
                )
    return manifest, states


def decision_function(matrix727: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Apply saved affine weights to full-schema inputs in the four-state order."""
    validate_candidate_state(state)
    if (
        not isinstance(matrix727, np.ndarray)
        or matrix727.dtype != np.dtype("float64")
        or matrix727.ndim != 2
        or matrix727.shape[1] != 727
        or matrix727.shape[0] == 0
        or not np.isfinite(matrix727).all()
    ):
        raise CandidateStateError(
            "router input must be a nonempty finite float64 (n, 727) matrix"
        )
    coefficients = np.asarray(state["coef"], dtype=np.float64)
    intercept = np.asarray(state["intercept"], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        scores = matrix727[:, state["feature_indices"]] @ coefficients.T + intercept
    if not np.isfinite(scores).all():
        raise CandidateStateError("router decision function produced nonfinite scores")
    return scores


def predict_router_probabilities(
    matrix727: np.ndarray, state: dict[str, Any]
) -> np.ndarray:
    """Return stable float64 probabilities ordered rescue/harm/correct/wrong."""
    scores = decision_function(matrix727, state)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        centered = scores - np.max(scores, axis=1, keepdims=True)
        weights = np.exp(centered)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise CandidateStateError("router probability computation failed")
    return probabilities


__all__ = [
    "SCHEMA_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "SCHEMA_MAX_ITER",
    "TARGET_ORDER",
    "C_GRID",
    "TRAINING_SEEDS",
    "VARIANTS",
    "FROZEN_FIT_PARAMETERS",
    "IMPLEMENTATION",
    "CandidateStateError",
    "full_feature_columns",
    "variant_feature_indices",
    "candidate_filename",
    "canonical_sha256",
    "candidate_state_sha256",
    "seal_candidate_state",
    "validate_candidate_state",
    "read_candidate_state",
    "load_candidate_set_for_inference",
    "decision_function",
    "predict_router_probabilities",
]
