#!/usr/bin/env python3
"""One training-only recovery of the frozen seed-1 image confidence parity failure.

This does not resume the paused study or change its implementation/recipe. It
repeats the same deterministic fit without supplying validation features, then
recovers the already-completed state from exception traceback locals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multimodal import geo_helpfulness_confidnet_oof_fit as fit

STUDY = ROOT / "multimodal_artifacts/analysis/cs/gse_100m/geo_helpfulness/confidnet_oof_v1"
OUTPUT = STUDY.parent / "confidnet_oof_v1_diagnostics"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    if OUTPUT.exists():
        raise RuntimeError("Diagnostic output already exists; repetition is prohibited")
    declaration_path = STUDY / "declaration.json"
    declaration = json.loads(declaration_path.read_text())
    for name, digest in declaration["file_sha256"].items():
        if name.endswith(".py") and sha(ROOT / name) != digest:
            raise RuntimeError(f"Declared source changed: {name}")
    train_path = STUDY / "features/seed_1/train.npz"
    protected = {str(path.relative_to(ROOT)): sha(path) for path in STUDY.rglob("*") if path.is_file()}
    inputs = {
        str(declaration_path.relative_to(ROOT)): sha(declaration_path),
        str(train_path.relative_to(ROOT)): sha(train_path),
        str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
        **{name: digest for name, digest in declaration["file_sha256"].items() if name.endswith(".py")},
    }
    OUTPUT.mkdir()
    write(OUTPUT / "diagnostic_declaration.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "one exact deterministic seed1/image training-only repetition to recover parity failure",
        "input_sha256": inputs,
        "training_seed": 1, "expert": "image",
        "parameters": declaration["config"]["confidence_parameters"],
        "validation_features_accessed": False, "validation_labels_accessed": False,
        "production_resumed": False,
        "protected_study_files_sha256": protected,
    })
    with np.load(train_path, allow_pickle=False) as arrays:
        hidden = arrays["image_hidden"].copy()
        logits = arrays["image_logits"].copy()
        labels = arrays["labels"].copy()
        row_uid = arrays["row_uid"].copy()
    started = time.perf_counter()
    try:
        fit.fit_confidence(hidden, logits, labels, {}, seed=1,
                           parameters=declaration["config"]["confidence_parameters"])
    except fit.ConfidNetOOFFitError as error:
        if str(error) != "train confidence portable parity exceeds 1e-06":
            raise
        trace = error.__traceback__
        captured = None
        while trace is not None:
            if trace.tb_frame.f_code.co_name == "fit_confidence" and Path(trace.tb_frame.f_code.co_filename).resolve() == Path(fit.__file__).resolve():
                captured = trace.tb_frame.f_locals
            trace = trace.tb_next
        if captured is None or captured["epoch"] != 500 or captured["name"] != "train":
            raise RuntimeError("Failure context is not the expected final training-parity check")
        state = captured["state"]
        history = captured["history"]
        metadata = captured["metadata"]
        native = captured["native"].copy()
        portable = captured["portable"].copy()
        exception_reason = str(error)
    else:
        raise RuntimeError("Frozen parity failure did not reproduce")
    write(OUTPUT / "recovered_state.json", state)
    write(OUTPUT / "recovered_history.json", history)
    write(OUTPUT / "recovered_fit_metadata.json", metadata)
    np.savez_compressed(OUTPUT / "training_predictions.npz", row_uid=row_uid, native=native, portable=portable)
    # Replay after a JSON roundtrip; no validation inputs are loaded anywhere.
    reread = json.loads((OUTPUT / "recovered_state.json").read_text())
    replay = fit.predict_confidence(reread, hidden, native=True)
    replay64 = fit.predict_confidence(reread, hidden, native=False)
    tcp, failures = fit._tcp(logits, labels, len(hidden))
    difference = np.abs(native.astype(np.float64) - portable)
    probabilities = [0, .25, .5, .75, .9, .95, .99, 1]
    mse = float(np.mean((native.astype(np.float64) - tcp) ** 2))
    summary = {
        "status": "recovered_expected_verification_failure",
        "failure_reason": exception_reason,
        "runtime_seconds": time.perf_counter() - started,
        "completed_epochs": len(history),
        "native_float32_replay_exact": bool(np.array_equal(native, replay)),
        "portable_float64_replay_exact": bool(np.array_equal(portable, replay64)),
        "maximum_native_replay_abs_error": float(np.max(np.abs(native.astype(np.float64)-replay))),
        "native_portable_maximum_abs_error": float(difference.max()),
        "native_portable_mean_abs_error": float(difference.mean()),
        "difference_quantiles": dict(zip(map(str, probabilities), np.quantile(difference, probabilities).tolist())),
        "rows_above_1e_minus6": int(np.count_nonzero(difference > 1e-6)),
        "rows_above_1e_minus5": int(np.count_nonzero(difference > 1e-5)),
        "maximum_difference_row_uid": str(row_uid[difference.argmax()]),
        "native_mse": mse,
        "recorded_final_epoch_mse": history[-1]["train_mse"],
        "mse_exact": mse == history[-1]["train_mse"],
        "training_errors": int(failures.sum()),
        "prediction_summary": fit._prediction_summary(native),
        "all_history_numeric_finite": all(np.isfinite(item[key]) for item in history for key in ("train_mse", "optimization_mse", "maximum_gradient_l2_norm", "maximum_absolute_gradient")),
        "zero_gradient_epochs": [item["epoch"] for item in history if item["maximum_absolute_gradient"] == 0],
        "final_gradient_l2_norm": history[-1]["maximum_gradient_l2_norm"],
        "final_maximum_absolute_gradient": history[-1]["maximum_absolute_gradient"],
        "validation_features_accessed": False,
        "validation_labels_accessed": False,
        "production_resumed": False,
    }
    current = {str(path.relative_to(ROOT)): sha(path) for path in STUDY.rglob("*") if path.is_file()}
    if current != protected:
        raise RuntimeError("Original study files changed during diagnostic")
    summary["protected_study_files_unchanged"] = True
    write(OUTPUT / "diagnostic_report.json", summary)
    write(OUTPUT / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
          "file_sha256": {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}})
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
