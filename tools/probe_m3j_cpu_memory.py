#!/usr/bin/env python3
"""Short, non-destructive user-space RAM probe for M3J verification failures.

Only this process's newly allocated memory is written. A pass provides limited
negative evidence for the allocated pages and duration; it does not establish
hardware health, test all RAM, or identify the cause of a model bit flip.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np


CHUNK_ELEMENTS = 1024 * 1024
PATTERNS = (
    ("zero", 0x0000000000000000),
    ("ones", 0xFFFFFFFFFFFFFFFF),
    ("alternating_aa", 0xAAAAAAAAAAAAAAAA),
    ("alternating_55", 0x5555555555555555),
    ("address_xor", None),
    ("inverse_address_xor", None),
)


def memory_info() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key + "_bytes"] = int(rest.split()[0]) * 1024
    return values


def expected_values(name: str, constant: int | None, start: int, stop: int):
    if constant is not None:
        return np.uint64(constant)
    values = np.arange(start, stop, dtype=np.uint64)
    values ^= np.uint64(0x9E3779B97F4A7C15)
    return ~values if name == "inverse_address_xor" else values


def mismatches(actual: np.ndarray, expected, start: int) -> tuple[int, list[dict]]:
    bad = np.flatnonzero(actual != expected)
    examples = []
    for local in bad[:8]:
        wanted = expected if np.isscalar(expected) else expected[local]
        value = int(actual[local])
        examples.append({"byte_offset": (start + int(local)) * 8,
                         "expected_hex": f"{int(wanted):016x}",
                         "observed_hex": f"{value:016x}",
                         "xor_hex": f"{value ^ int(wanted):016x}"})
    return len(bad), examples


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run(gib: int, repeats: int, seconds: float, output: Path) -> dict:
    start_time = time.monotonic()
    size = gib * 1024 ** 3
    info = memory_info()
    if info["MemAvailable_bytes"] < size * 2:
        raise RuntimeError("Insufficient available RAM for a bounded, low-impact probe")
    positive = np.arange(16, dtype=np.uint64)
    reference = positive.copy()
    positive[7] ^= np.uint64(0x40)
    count, examples = mismatches(positive, reference, 0)
    if count != 1 or examples[0]["byte_offset"] != 56 or examples[0]["xor_hex"] != "0000000000000040":
        raise RuntimeError("Probe mismatch detector failed its injected-bit positive control")
    cpu_name = next((line.split(":", 1)[1].strip()
                     for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown")
    report = {
        "schema": "m3j_cpu_memory_probe.v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running", "allocated_bytes": size, "requested_repeats": repeats,
        "time_budget_seconds": seconds, "chunk_bytes": CHUNK_ELEMENTS * 8,
        "cpu_model": cpu_name, "python": platform.python_version(), "numpy": np.__version__,
        "memory_before": info, "detector_positive_control": "passed",
        "system_settings_changed": False, "allocated_pages_only": True,
        "limitation": "Short user-space probe of one allocated region; a pass does not establish hardware health, test all RAM or GPUs, or diagnose the prior model corruption.",
        "completed_patterns": [], "mismatching_uint64_values": 0, "error_examples": [],
    }
    write_report(output, report)
    buffer = np.empty(size // 8, dtype=np.uint64)
    deadline = start_time + seconds
    timeout = False
    try:
        for repeat in range(1, repeats + 1):
            for name, constant in PATTERNS:
                pattern_started = time.monotonic()
                for offset in range(0, len(buffer), CHUNK_ELEMENTS):
                    if time.monotonic() >= deadline:
                        timeout = True
                        break
                    end = min(offset + CHUNK_ELEMENTS, len(buffer))
                    buffer[offset:end] = expected_values(name, constant, offset, end)
                if timeout:
                    break
                checked_bytes = 0
                for offset in range(0, len(buffer), CHUNK_ELEMENTS):
                    if time.monotonic() >= deadline:
                        timeout = True
                        break
                    end = min(offset + CHUNK_ELEMENTS, len(buffer))
                    count, examples = mismatches(buffer[offset:end], expected_values(name, constant, offset, end), offset)
                    checked_bytes += (end - offset) * 8
                    if count:
                        report["mismatching_uint64_values"] += count
                        report["error_examples"].extend(dict(example, repeat=repeat, pattern=name)
                                                        for example in examples)
                        break
                record = {"repeat": repeat, "pattern": name, "checked_bytes": checked_bytes,
                          "completed": checked_bytes == size, "elapsed_seconds": time.monotonic() - pattern_started}
                report["completed_patterns"].append(record)
                report["elapsed_seconds"] = time.monotonic() - start_time
                write_report(output, report)
                print(json.dumps(record), flush=True)
                if timeout or report["mismatching_uint64_values"]:
                    break
            if timeout or report["mismatching_uint64_values"]:
                break
    finally:
        del buffer
    report["status"] = ("errors_detected" if report["mismatching_uint64_values"]
                        else "time_budget_reached" if timeout else "no_errors_observed")
    report["elapsed_seconds"] = time.monotonic() - start_time
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["memory_after"] = memory_info()
    write_report(output, report)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "mismatching_uint64_values": report["mismatching_uint64_values"],
                      "report": str(output.resolve())}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=110)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.gib <= 4 and 1 <= args.repeats <= 3 and 1 <= args.seconds <= 110):
        parser.error("Use 1–4 GiB, 1–3 repeats, and at most 110 seconds")
    run(args.gib, args.repeats, args.seconds, args.output)


if __name__ == "__main__":
    main()
