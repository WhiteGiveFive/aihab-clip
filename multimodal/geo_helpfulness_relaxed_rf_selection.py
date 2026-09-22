"""Development-only guardrail sensitivity for the already selected RF routers.

The caller authenticates parent manifests and model bytes before calling this
pure module. Model/configuration identities are inherited; no fitting, model
selection, test evaluation, or file access is available here.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import PurePosixPath
import re
from typing import Any

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_relaxed_guardrails_selection as shared
from multimodal import geo_helpfulness_router_selection as original
from multimodal.geo_helpfulness_router_candidate_state import canonical_sha256
from multimodal.geo_helpfulness_router_heuristics import apply_heuristic_threshold


STUDY_ID = "relaxed_guardrails_rf_v1"
SCHEMA_VERSION = "geo_helpfulness.relaxed_rf_selection.v1"
VARIANTS = original.VARIANTS
GUARDRAILS = deepcopy(shared.GUARDRAILS)
RouterSelectionError = original.RouterSelectionError
CAPACITY_GROUPS = frozenset(
    f"{family}__{variant}" for family in ("lr", "rf", "xgb") for variant in VARIANTS
)
SOURCE_POLICY_IDS = (
    frozenset((*VARIANTS, *original.HEURISTICS))
    | frozenset(
        f"capacity__{family}__{variant}"
        for family in ("rf", "xgb")
        for variant in VARIANTS
    )
    | frozenset(f"mlp__{variant}" for variant in VARIANTS)
)


def policy_id(variant: str, guardrail_id: str) -> str:
    """Return one of the 27 predeclared RF sensitivity identities."""
    name = shared.policy_id(variant, guardrail_id)
    return name.replace("relaxed__", "relaxed__rf__", 1)


def select_common_threshold(
    rows: pd.DataFrame, *, minimum_plots: int, harm_numerator: int
) -> dict[str, Any]:
    """Apply the same exact guardrails, candidate boundaries and ties as LR."""
    return shared.select_common_threshold(
        rows, minimum_plots=minimum_plots, harm_numerator=harm_numerator
    )


def _validate_source_metadata(original_selection, frozen_policies):
    if (
        not isinstance(original_selection, dict)
        or not isinstance(original_selection.get("groups"), dict)
        or set(original_selection["groups"]) != CAPACITY_GROUPS
        or original_selection.get("training_seeds") != list(original.SEEDS)
        or original_selection.get("test_data_read_by_scorer") is not False
        or original_selection.get("fitting_performed_by_scorer") is not False
    ):
        raise RouterSelectionError(
            "expected the original capacity development selection"
        )
    if (
        not isinstance(frozen_policies, dict)
        or set(frozen_policies) != SOURCE_POLICY_IDS
    ):
        raise RouterSelectionError("expected the 17 frozen legacy policy declarations")
    for variant in VARIANTS:
        group_id = f"rf__{variant}"
        source_id = f"capacity__{group_id}"
        chosen = original_selection["groups"][group_id]
        frozen = frozen_policies[source_id]
        if (
            not isinstance(chosen, dict)
            or chosen.get("group_id") != group_id
            or chosen.get("family") != "rf"
            or chosen.get("variant") != variant
            or chosen.get("selected_policy_id") != source_id
            or not isinstance(chosen.get("selected_config_id"), str)
            or not chosen["selected_config_id"].startswith(group_id + "__")
            or not isinstance(chosen.get("threshold"), dict)
        ):
            raise RouterSelectionError(
                "original RF configuration identity is inconsistent"
            )
        config_id = chosen["selected_config_id"]
        if (
            not isinstance(frozen, dict)
            or frozen.get("learner") != "rf"
            or frozen.get("variant_id") != variant
            or frozen.get("config_id") != config_id
            or frozen.get("C_hex") != ""
            or frozen.get("threshold_specification")
            != chosen["threshold"].get("threshold_specification")
        ):
            raise RouterSelectionError(
                "frozen RF policy differs from original selection"
            )
        models = frozen.get("models")
        if not isinstance(models, dict) or set(models) != {
            str(s) for s in original.SEEDS
        }:
            raise RouterSelectionError("frozen RF policy requires all four model seeds")
        for seed, record in models.items():
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or PurePosixPath(record["path"]).name
                != f"{config_id}__seed_{seed}.json"
                or any(
                    not isinstance(record.get(key), str)
                    or re.fullmatch(r"[0-9a-f]{64}", record[key]) is None
                    for key in ("file_sha256", "record_sha256", "state_sha256")
                )
            ):
                raise RouterSelectionError(
                    "frozen RF model identity/hash metadata is invalid"
                )


def build_selection(
    original_selection: dict[str, Any],
    validation_actions: pd.DataFrame,
    frozen_policies: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the three original RF policies, then select all 27 thresholds.

    ``original_selection`` and ``validation_actions`` come from the sealed
    capacity development score bundle. ``frozen_policies`` is the flat 17-policy
    mapping in the authenticated legacy declaration. All source RF baselines
    must reproduce before any relaxed guardrail is evaluated.
    """
    _validate_source_metadata(original_selection, frozen_policies)
    required = {
        "policy_id",
        "row_uid",
        "plot_idx",
        "training_seed",
        "target_state",
        "raw_pred",
        "geo_pred",
        "score",
        "acted",
        "threshold_specification",
    }
    rf_ids = {f"capacity__rf__{variant}" for variant in VARIANTS}
    allowed_ids = {f"capacity__{group}" for group in CAPACITY_GROUPS} | {
        VARIANTS[0],
        *original.HEURISTICS,
    }
    if (
        not isinstance(validation_actions, pd.DataFrame)
        or validation_actions.columns.duplicated().any()
        or not required.issubset(validation_actions)
        or validation_actions.policy_id.isna().any()
        or not rf_ids.issubset(validation_actions.policy_id)
        or not set(validation_actions.policy_id).issubset(allowed_ids)
    ):
        raise RouterSelectionError(
            "validation actions require the three original RF policies"
        )
    prepared, reference = {}, None
    for variant in VARIANTS:
        group_id = f"rf__{variant}"
        source_id = f"capacity__{group_id}"
        chosen = original_selection["groups"][group_id]
        saved = chosen["threshold"]
        source = validation_actions.loc[validation_actions.policy_id.eq(source_id)]
        source = source.sort_values(["row_uid", "training_seed"], kind="stable")
        rows = original._threshold_rows(source)
        common = rows.drop(columns="score")
        if reference is not None and not common.equals(reference):
            raise RouterSelectionError("validation targets differ across RF variants")
        reference = common
        baseline = original.select_common_threshold(rows)
        if baseline != saved:
            raise RouterSelectionError(
                f"original RF threshold curve does not reproduce: {variant}"
            )
        expected = apply_heuristic_threshold(
            rows.score.to_numpy(),
            rows.raw_pred.to_numpy(),
            rows.geo_pred.to_numpy(),
            saved["threshold_specification"],
        )
        if source.acted.dtype != np.dtype("bool") or not np.array_equal(
            source.acted.to_numpy(), expected
        ):
            raise RouterSelectionError("original RF action mask does not reproduce")
        for value in source.threshold_specification.unique():
            try:
                specification = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise RouterSelectionError("invalid saved RF action threshold") from exc
            if specification != saved["threshold_specification"]:
                raise RouterSelectionError(
                    "RF action threshold differs from original selection"
                )
        prepared[variant] = (rows, baseline, chosen, frozen_policies[source_id])
    result = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "original_selection_sha256": canonical_sha256(original_selection),
        "original_frozen_policies_sha256": canonical_sha256(frozen_policies),
        "guardrails": deepcopy(list(GUARDRAILS)),
        "policies": {},
        "original_baseline_parity": {},
    }
    for variant, (rows, baseline, chosen, frozen) in prepared.items():
        for guardrail in GUARDRAILS:
            threshold = shared._select_from_curve(rows, baseline, guardrail)
            if guardrail["minimum_plots"] == 20 and guardrail["harm_numerator"] == 5:
                if {
                    k: v for k, v in threshold.items() if k != "guardrails"
                } != baseline:
                    raise RouterSelectionError(
                        "relaxed RF baseline differs from original"
                    )
                result["original_baseline_parity"][variant] = True
            name = policy_id(variant, guardrail["guardrail_id"])
            result["policies"][name] = {
                "learner": "rf",
                "variant": variant,
                "config_id": chosen["selected_config_id"],
                "original_group_id": chosen["group_id"],
                "original_policy_id": chosen["selected_policy_id"],
                "guardrail_id": guardrail["guardrail_id"],
                "models": deepcopy(frozen["models"]),
                "source_configuration_selection": deepcopy(
                    {k: v for k, v in chosen.items() if k != "threshold"}
                ),
                "threshold": threshold,
            }
    result["selection_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "STUDY_ID",
    "SCHEMA_VERSION",
    "VARIANTS",
    "GUARDRAILS",
    "RouterSelectionError",
    "policy_id",
    "select_common_threshold",
    "build_selection",
]
