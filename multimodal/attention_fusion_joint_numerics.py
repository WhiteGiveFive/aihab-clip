"""Audited numerical comparisons for explicitly exploratory M3J execution.

The exploratory bounds are engineering choices, not a guarantee that a machine
with memory errors is reliable. File authentication belongs to the checkpoint
store and must remain exact. This module never repairs values or moves a frozen
reference to an observed value.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


_LIMITS = dict(
    frozen_atol=1e-8,
    frozen_rtol=1e-5,
    max_frozen_changed_coordinates=8,
    max_frozen_unique_deviations=16,
    recomputed_atol=1e-6,
    recomputed_rtol=1e-5,
)
STRICT_POLICY = dict(mode="strict", **_LIMITS)
EXPLORATORY_POLICY = dict(mode="exploratory", **_LIMITS)
_SCHEMA = "attention_fusion_joint.numerical_audit.v1"
_CHUNK_ELEMENTS = 262144
_MAX_EVIDENCE = 32


def normalize_numerical_policy(policy: dict | None = None) -> dict:
    """Require explicit opt-in and reject silently altered numerical budgets."""
    if policy is None:
        return dict(STRICT_POLICY)
    if not isinstance(policy, dict) or set(policy) - set(STRICT_POLICY):
        raise ValueError("Unknown numerical policy fields")
    mode = policy.get("mode", "strict")
    if mode not in ("strict", "exploratory"):
        raise ValueError("Numerical policy mode must be strict or exploratory")
    result = dict(mode=mode, **_LIMITS)
    for key, expected in _LIMITS.items():
        value = policy.get(key, expected)
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"Numerical policy changes the declared bound: {key}")
    return result


def _json_key(*values: Any) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _audit_digest(state: dict) -> str:
    payload = {key: value for key, value in state.items() if key != "audit_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _validate_audit_state(state: dict, policy: dict | None = None) -> dict:
    if not isinstance(state, dict) or state.get("schema_version") != _SCHEMA:
        raise ValueError("Numerical audit schema or policy changed on resume")
    normalized = normalize_numerical_policy(state.get("policy"))
    if state.get("policy") != normalized or policy is not None and normalized != policy:
        raise ValueError("Numerical audit schema or policy changed on resume")
    if state.get("audit_sha256") != _audit_digest(state):
        raise ValueError("Saved numerical audit checksum differs")
    for key in ("canonical_references", "changed_coordinates", "unique_deviations"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"Malformed numerical audit: {key}")
    if not isinstance(state.get("events"), list):
        raise ValueError("Malformed numerical audit events")
    if len(state["changed_coordinates"]) > normalized["max_frozen_changed_coordinates"] \
            or len(state["unique_deviations"]) > normalized["max_frozen_unique_deviations"]:
        raise ValueError("Saved numerical audit exceeds the declared frozen budget")
    return deepcopy(state)


def _number(value: Any) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(value) else repr(value)
    return repr(value)


def _info(value: Any) -> dict | None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError("Numerical comparison requires dense, unquantized tensors")
        return dict(backend="torch", dtype=str(value.dtype), shape=list(value.shape),
                    flat=value.detach().reshape(-1), count=value.numel(),
                    itemsize=value.element_size(), floating=value.is_floating_point(),
                    complex=value.is_complex())
    if isinstance(value, (np.ndarray, np.generic)) or type(value) is float:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays are not supported by numerical auditing")
        backend = "python_float" if type(value) is float else (
            "numpy_scalar" if isinstance(value, np.generic) else "numpy")
        return dict(backend=backend, dtype=array.dtype.str, shape=list(array.shape),
                    flat=array.reshape(-1), count=array.size, itemsize=array.dtype.itemsize,
                    floating=array.dtype.kind == "f", complex=array.dtype.kind == "c")
    return None


def _chunk(info: dict, start: int, stop: int) -> tuple[bytes, np.ndarray | None]:
    value = info["flat"][start:stop]
    if info["backend"] == "torch":
        value = value.cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        numeric = value.to(torch.complex128 if info["complex"] else torch.float64).numpy() \
            if info["floating"] or info["complex"] else None
    else:
        value = np.ascontiguousarray(value)
        raw = value.tobytes()
        numeric = value.astype(np.complex128 if info["complex"] else np.float64) \
            if info["floating"] or info["complex"] else None
    return raw, numeric


class NumericalAudit:
    """Compare trees, persist evidence, and enforce cumulative frozen budgets.

    ``namespace`` and tree paths identify a canonical frozen tensor throughout a
    fit. Use the same namespace for save/load/epoch checks of those same tensors.
    ``copy`` comparisons remain exact, including under exploratory policy.
    """

    def __init__(self, policy: dict | None = None, path: str | Path | None = None):
        self.policy = normalize_numerical_policy(policy)
        self.path = Path(path) if path is not None else None
        self._restored_from_snapshot = False
        self._state = dict(schema_version=_SCHEMA, policy=self.policy,
                           canonical_references={}, changed_coordinates={},
                           unique_deviations={}, events=[])
        if self.path is not None and self.path.exists():
            self._state = _validate_audit_state(json.loads(self.path.read_text()), self.policy)

    @classmethod
    def from_snapshot(cls, snapshot: dict, path: str | Path | None = None) -> "NumericalAudit":
        """Restore committed evidence for portable inference into a fresh audit.

        The caller must obtain the snapshot from an authenticated checkpoint or
        export manifest. A self-checksum detects accidental changes, not rollback;
        ordinary resume must instead call ``require_committed_prefix``.
        """
        state = _validate_audit_state(snapshot)
        if path is not None and Path(path).exists():
            raise ValueError("Snapshot restore requires a fresh numerical audit path")
        audit = cls(state["policy"], path)
        audit._state = state
        audit._restored_from_snapshot = True
        return audit

    @property
    def snapshot(self) -> dict:
        """Return independent complete evidence to embed in an authenticated manifest."""
        state = deepcopy(self._state)
        state["summary"] = self.summary
        state["audit_sha256"] = _audit_digest(state)
        return state

    def require_committed_prefix(self, snapshot: dict) -> None:
        """Reject missing/rolled-back evidence while allowing interrupted later events."""
        committed = _validate_audit_state(snapshot, self.policy)
        current = _validate_audit_state(self.snapshot, self.policy)
        if self.path is not None and not self.path.exists() and not self._restored_from_snapshot:
            raise ValueError("Numerical audit is missing for the committed checkpoint")
        prefix = committed["events"]
        if len(current["events"]) < len(prefix) or current["events"][:len(prefix)] != prefix:
            raise ValueError("Numerical audit events are older than or differ from the committed checkpoint")
        for key in ("canonical_references", "changed_coordinates", "unique_deviations"):
            if any(name not in current[key] or current[key][name] != value
                   for name, value in committed[key].items()):
                raise ValueError(f"Numerical audit {key} lost or changed committed evidence")

    @property
    def summary(self) -> dict:
        events = self._state["events"]
        relative = [e["max_relative_error"] for e in events if e.get("max_relative_error") is not None]
        return dict(
            mode=self.policy["mode"], comparisons=len(events),
            accepted_numerical_comparisons=sum(e["status"] == "accepted_tolerance" for e in events),
            rejected_comparisons=sum(e["status"] == "rejected" for e in events),
            frozen_changed_coordinates=len(self._state["changed_coordinates"]),
            frozen_unique_deviations=len(self._state["unique_deviations"]),
            max_absolute_error=max((e.get("max_absolute_error", 0.0) for e in events), default=0.0),
            max_relative_error=max(relative, default=0.0),
            unbounded_relative_error=any(e.get("max_relative_error") is None for e in events),
            integrity_deviations_detected=any(
                e["kind"] in ("frozen", "copy") and not e["exact"] for e in events),
        )

    def _save(self) -> None:
        self._state["summary"] = self.summary
        self._state["audit_sha256"] = _audit_digest(self._state)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            with temporary.open("w") as handle:
                json.dump(self._state, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)

    def compare(self, actual: Any, expected: Any, context: str,
                kind: str = "frozen", namespace: str = "default") -> dict:
        if kind not in ("frozen", "recomputed", "copy"):
            raise ValueError("Unknown numerical comparison kind")
        if not isinstance(context, str) or not isinstance(namespace, str) or not namespace:
            raise ValueError("Numerical comparison needs a context and stable namespace")
        report = dict(context=context, kind=kind, namespace=namespace, status="exact", exact=True,
                      differing_elements=0, max_absolute_error=0.0, max_relative_error=0.0, tensors=[])
        staged_coordinates = {}
        staged_deviations = {}
        failure = []

        def visit(left: Any, right: Any, path: str) -> None:
            left_info, right_info = _info(left), _info(right)
            if left_info is not None or right_info is not None:
                if left_info is None or right_info is None:
                    raise ValueError(f"Numerical value type differs at {path}")
                tensor, reasons = self._array(left_info, right_info, path, kind, namespace,
                                              staged_coordinates, staged_deviations)
                if not tensor["exact"] or reasons:
                    report["tensors"].append(tensor)
                report["exact"] = report["exact"] and tensor["exact"]
                report["differing_elements"] += tensor["differing_elements"]
                report["max_absolute_error"] = max(report["max_absolute_error"], tensor["max_absolute_error"])
                if tensor["max_relative_error"] is None:
                    report["max_relative_error"] = None
                elif report["max_relative_error"] is not None:
                    report["max_relative_error"] = max(report["max_relative_error"], tensor["max_relative_error"])
                failure.extend(reasons)
                return
            if type(left) is not type(right):
                raise ValueError(f"Tree value type differs at {path}")
            if isinstance(right, dict):
                if left.keys() != right.keys():
                    raise ValueError(f"Tree keys differ at {path}")
                for key in right:
                    if type(key) not in (str, int):
                        raise ValueError(f"Unsupported dictionary key at {path}")
                    visit(left[key], right[key], f"{path}[{key!r}]")
            elif isinstance(right, (list, tuple)):
                if len(left) != len(right):
                    raise ValueError(f"Tree lengths differ at {path}")
                for index, (item, reference) in enumerate(zip(left, right)):
                    visit(item, reference, f"{path}[{index}]")
            elif type(right) in (str, int, bool, bytes, type(None)):
                if left != right:
                    raise ValueError(f"Nonfloating value differs at {path}")
            else:
                raise ValueError(f"Unsupported tree value at {path}: {type(right).__name__}")

        try:
            visit(actual, expected, "$")
            if kind == "frozen":
                count = len(self._state["changed_coordinates"].keys() | staged_coordinates.keys())
                deviations = len(self._state["unique_deviations"].keys() | staged_deviations.keys())
                if count > self.policy["max_frozen_changed_coordinates"]:
                    failure.append("Cumulative frozen changed-coordinate budget exceeded")
                if deviations > self.policy["max_frozen_unique_deviations"]:
                    failure.append("Cumulative frozen unique-deviation budget exceeded")
            if failure:
                raise ValueError("; ".join(dict.fromkeys(failure)))
        except ValueError as error:
            report.update(status="rejected", exact=False, reason=str(error))
            self._state["events"].append(report)
            self._save()
            raise ValueError(f"Numerical audit rejected {context}: {error}") from error
        if not report["exact"]:
            report["status"] = "accepted_tolerance"
        for key, entry in staged_coordinates.items():
            self._state["changed_coordinates"].setdefault(key, entry)
        for key, entry in staged_deviations.items():
            self._state["unique_deviations"].setdefault(key, entry)
        self._state["events"].append(report)
        self._save()
        return deepcopy(report)

    def _array(self, actual: dict, expected: dict, path: str, kind: str, namespace: str,
               coordinates: dict, deviations: dict) -> tuple[dict, list[str]]:
        metadata = ("backend", "dtype", "shape", "count", "itemsize")
        if any(actual[key] != expected[key] for key in metadata):
            raise ValueError(f"Array type, shape or dtype differs at {path}")
        description = {key: expected[key] for key in ("backend", "dtype", "shape")}
        seed = json.dumps(description, sort_keys=True).encode()
        canonical_hash, observed_hash = hashlib.sha256(seed), hashlib.sha256(seed)
        result = dict(path=path, **description, exact=True, differing_elements=0,
                      max_absolute_error=0.0, max_relative_error=0.0, changed_elements=[])
        reasons = []
        exploratory = self.policy["mode"] == "exploratory" and kind != "copy"
        for start in range(0, expected["count"], _CHUNK_ELEMENTS):
            stop = min(start + _CHUNK_ELEMENTS, expected["count"])
            observed_bytes, left = _chunk(actual, start, stop)
            canonical_bytes, right = _chunk(expected, start, stop)
            observed_hash.update(observed_bytes)
            canonical_hash.update(canonical_bytes)
            finite = left is None or bool(np.isfinite(left).all() and np.isfinite(right).all())
            if not finite:
                reasons.append(f"Nonfinite values at {path}")
            if observed_bytes == canonical_bytes:
                continue
            result["exact"] = False
            itemsize = expected["itemsize"]
            byte_delta = np.frombuffer(observed_bytes, dtype=np.uint8) != np.frombuffer(canonical_bytes, dtype=np.uint8)
            changed = np.flatnonzero(byte_delta.reshape(-1, itemsize).any(axis=1))
            result["differing_elements"] += len(changed)
            if not exploratory or not expected["floating"]:
                reasons.append(f"Byte-exact comparison failed at {path}")
            if expected["floating"] and finite:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    absolute = np.abs(left[changed] - right[changed])
                    denominator = np.abs(right[changed])
                    relative = np.divide(absolute, denominator, out=np.full_like(absolute, np.inf),
                                         where=denominator != 0)
                    relative[(absolute == 0) & (denominator == 0)] = 0
                maximum = float(absolute.max(initial=0))
                if not math.isfinite(maximum):
                    reasons.append(f"Nonfinite difference at {path}")
                else:
                    result["max_absolute_error"] = max(result["max_absolute_error"], maximum)
                if np.isfinite(relative).all() and result["max_relative_error"] is not None:
                    result["max_relative_error"] = max(result["max_relative_error"], float(relative.max(initial=0)))
                else:
                    result["max_relative_error"] = None
                if exploratory:
                    if kind == "frozen":
                        accepted = ((denominator != 0)
                                    & (absolute <= self.policy["frozen_atol"])
                                    & (relative <= self.policy["frozen_rtol"]))
                    else:
                        accepted = absolute <= (self.policy["recomputed_atol"] + self.policy["recomputed_rtol"] * denominator)
                    if not bool(accepted.all()):
                        reasons.append(f"Declared {kind} numerical bound exceeded at {path}")
            # Evidence and candidate budget sets are bounded even for huge bad tensors.
            for local in changed[:_MAX_EVIDENCE]:
                index = start + int(local)
                coordinate = _json_key(namespace, path, index)
                offset = int(local) * itemsize
                observed_hex = observed_bytes[offset:offset + itemsize].hex()
                canonical_hex = canonical_bytes[offset:offset + itemsize].hex()
                item = dict(flat_index=index, index=list(np.unravel_index(index, tuple(expected["shape"])))
                            if expected["shape"] else [], expected_bytes=canonical_hex,
                            actual_bytes=observed_hex)
                item["index"] = [int(v) for v in item["index"]]
                if left is not None:
                    item.update(expected=_number(right[local]), actual=_number(left[local]))
                else:
                    reference_value = expected["flat"][index]
                    observed_value = actual["flat"][index]
                    if isinstance(reference_value, torch.Tensor):
                        reference_value, observed_value = reference_value.item(), observed_value.item()
                    item.update(expected=_number(reference_value), actual=_number(observed_value))
                if len(result["changed_elements"]) < _MAX_EVIDENCE:
                    result["changed_elements"].append(item)
                if kind == "frozen":
                    entry = dict(namespace=namespace, path=path, **item)
                    if len(coordinates) <= self.policy["max_frozen_changed_coordinates"]:
                        coordinates.setdefault(coordinate, entry)
                    signature = _json_key(coordinate, observed_hex)
                    if len(deviations) <= self.policy["max_frozen_unique_deviations"]:
                        deviations.setdefault(signature, entry)
        result.update(canonical_sha256=canonical_hash.hexdigest(), observed_sha256=observed_hash.hexdigest(),
                      evidence_truncated=result["differing_elements"] > len(result["changed_elements"]))
        if kind == "frozen":
            key = _json_key(namespace, path)
            canonical = dict(**description, sha256=result["canonical_sha256"])
            previous = self._state["canonical_references"].get(key)
            if previous is not None and previous != canonical:
                reasons.append(f"Canonical frozen reference changed at {path}")
            elif previous is None:
                self._state["canonical_references"][key] = canonical
        return result, reasons
