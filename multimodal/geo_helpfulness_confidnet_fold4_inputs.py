"""Fold-four-only calibration and numerical inputs for the fixed-producer study.

No files or labels are discovered here. Both expert payloads must come from the
same fold-three producer; only the supplied held-out training rows fit scalar
temperatures and population moments. Confidence supervision remains native T=1.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from multimodal import geo_helpfulness_confidnet_fold4_extract as extraction
from multimodal import geo_helpfulness_router_numeric as numeric
from multimodal import geo_helpfulness_targets_features as features
from multimodal.geo_helpfulness_protocol import canonical_sha256


MODE_PREFIXES = {
    "image_only": "image", "geo_only": "geo", "raw_concat": "raw",
}
FEATURE_COLUMNS = tuple(numeric.output_feature_columns()[:25])


class ConfidNetFold4InputError(ValueError):
    """Inputs or saved preprocessing do not match the fixed-fold contract."""


def _fit_identity(train: Mapping[str, Any], seed: int) -> str:
    return canonical_sha256([
        {"row_uid": str(uid), "training_seed": seed} for uid in train["row_uid"]
    ])


def _check_state_identity(state: Mapping[str, Any], seed: int, count: int,
                          identity: str) -> None:
    if (state["seed"], state["fit_row_count"], state["fit_row_identity_sha256"]) != (
        seed, count, identity
    ):
        raise ConfidNetFold4InputError("saved preprocessing fit identity differs from fold 4")


def _check_temperature_replay(state: Mapping[str, Any], logits: np.ndarray,
                              labels: np.ndarray, mode: str) -> None:
    if state["mode"] != mode:
        raise ConfidNetFold4InputError("saved temperature expert mode differs")
    for field, temperature in (("native_nll", 1.0),
                               ("calibrated_nll", state["temperature"])):
        observed = numeric.multiclass_nll(logits, labels, temperature)
        if observed != state[field]:
            raise ConfidNetFold4InputError("saved temperature NLL differs from fold-4 supervision")


def _check_transform_replay(state: Mapping[str, Any], semantic: Any) -> None:
    matrix = semantic.loc[:, list(numeric.NUMERIC_INPUT_COLUMNS)].to_numpy(dtype=np.float64)
    for field, observed in (("mean", matrix.mean(axis=0)),
                            ("variance", matrix.var(axis=0, ddof=0))):
        if not np.array_equal(np.asarray(state[field]), observed):
            raise ConfidNetFold4InputError("saved transform moments differ from fold-4 features")


def build_inputs(train: Mapping[str, Any], validation: Mapping[str, Any], *,
                 seed: int, states: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit on fold 4, or replay supplied states without any optimization.

    The artifact caller authenticates payload bytes and exact M1 membership.
    This pure layer checks split provenance, label isolation, fitting identities,
    and saved numerical summaries. The production split has 852 training rows;
    smaller synthetic payloads are supported for validation tests.
    """

    if type(seed) is not int or seed not in features.TRAINING_SEEDS:
        raise ConfidNetFold4InputError("seed must be one of 1, 2, 3, 4")
    extraction.validate_split_payload(train, "train")
    extraction.validate_split_payload(validation, "validation")
    for name in ("row_uid", "plot_idx"):
        if set(train[name]) & set(validation[name]):
            raise ConfidNetFold4InputError(f"training and validation {name} overlap")
    fitting = states is None
    if not fitting:
        if not isinstance(states, Mapping) or set(states) != {"temperatures", "transform"}:
            raise ConfidNetFold4InputError("saved preprocessing states have unexpected keys")
        if not isinstance(states["temperatures"], Mapping) or set(
            states["temperatures"]
        ) != set(MODE_PREFIXES):
            raise ConfidNetFold4InputError("saved temperature mode set differs")
    identity = _fit_identity(train, seed)
    count = len(train["row_uid"])
    temperatures, probabilities = {}, {"train": [], "validation": []}
    labels = np.asarray(train["labels"], dtype=np.int64)
    for mode, prefix in MODE_PREFIXES.items():
        logits = train[f"{prefix}_logits"]
        if fitting:
            state = numeric.fit_expert_temperature(
                logits, labels, seed=seed, mode=mode,
                calibration_spec=numeric.CALIBRATION_SPEC,
                fit_row_identity_sha256=identity,
            )
        else:
            state = deepcopy(states["temperatures"][mode])
            numeric.validate_temperature_state(state)
            _check_state_identity(state, seed, count, identity)
            _check_temperature_replay(state, logits, labels, mode)
        temperatures[mode] = state
        for split, payload in (("train", train), ("validation", validation)):
            probabilities[split].append(numeric.apply_expert_temperature(
                payload[f"{prefix}_logits"], state
            ))
    semantic = {
        split: features.build_router_feature_frame(
            *values, probability_basis=features.CALIBRATED_PROBABILITY_BASIS
        ) for split, values in probabilities.items()
    }
    if fitting:
        transform = numeric.fit_router_feature_transform(
            semantic["train"], seed=seed, frozen_schema=features.build_router_feature_schema(),
            fit_row_identity_sha256=identity,
        )
    else:
        transform = deepcopy(states["transform"])
        numeric.validate_feature_transform_state(transform)
        _check_state_identity(transform, seed, count, identity)
        _check_transform_replay(transform, semantic["train"])
    target_names = features.derive_router_target_states(
        np.argmax(train["raw_logits"], axis=1),
        np.argmax(train["geo_logits"], axis=1), labels,
    )
    target_indices = {name: i for i, name in enumerate(features.TARGET_ORDER)}
    arrays = {
        "X_train": np.ascontiguousarray(numeric.transform_router_features(
            semantic["train"], transform
        )[:, :25]),
        "y_train": np.asarray([target_indices[name] for name in target_names], dtype=np.int64),
        "X_validation": np.ascontiguousarray(numeric.transform_router_features(
            semantic["validation"], transform
        )[:, :25]),
    }
    for split, payload in (("train", train), ("validation", validation)):
        for name in ("row_uid", "plot_idx"):
            arrays[f"{split}_{name}"] = payload[name].copy()
    return {
        "arrays": arrays,
        "states": {"temperatures": temperatures, "transform": transform},
        "metadata": {
            "schema_version": "geo_helpfulness.confidnet_fold4_inputs.v1",
            "training_seed": seed,
            "producer_fold": extraction.PRODUCER_FOLD,
            "same_expert_bundle_for_both_splits": True,
            "feature_columns": list(FEATURE_COLUMNS),
            "numeric_input_columns": list(numeric.NUMERIC_INPUT_COLUMNS),
            "target_order": list(features.TARGET_ORDER),
            "fit_row_identity_sha256": identity,
            "fit_row_count": count,
            "fit_plot_count": len(set(train["plot_idx"].tolist())),
            "validation_row_count": len(validation["row_uid"]),
            "validation_plot_count": len(set(validation["plot_idx"].tolist())),
            "calibration_fit_scope": "heldout_development_train_fold_4_only",
            "transform_fit_scope": "heldout_development_train_fold_4_only",
            "validation_labels_accessed": False,
            "validation_features_used_for_fitting": False,
            "tcp_supervision_probability_basis": "native_t1_uncalibrated",
        },
    }
