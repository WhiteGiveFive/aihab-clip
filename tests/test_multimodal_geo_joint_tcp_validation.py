"""Regression for sorted JSON versus original report-rendering order."""

from copy import deepcopy
import json

import pytest

from multimodal.geo_helpfulness_joint_tcp_report import format_report
from multimodal.geo_helpfulness_joint_tcp_validation import validate_report_outputs


def reports():
    mean = {"state": {"nll": .6, "brier": .4, "accuracy": .7, "macro_f1": .5},
            "tcp": {name: {"tcp_regression": {"mse": .1}} for name in ("fusion", "geo")}}
    diagnostic = {"splits": {split: {"mean_seed": deepcopy(mean)} for split in ("train", "validation")}}
    report = {
        "study_id": "joint_tcp_router_v1", "model_count": 64,
        "diagnostics": {name: deepcopy(diagnostic) for name in ("numeric25", "no_pairs79")},
        "validation_policies": {}, "validation_bootstrap": {"contrasts": {}},
    }
    result = {"report": report, "selections": {"z": {"threshold": 0}, "a": {"threshold": .1}}}
    saved = json.loads(json.dumps(report, sort_keys=True))
    selections = json.loads(json.dumps(result["selections"], sort_keys=True))
    return result, saved, selections, format_report(report)


def test_numeric_json_equality_does_not_preserve_original_markdown_order():
    result, saved, selections, markdown = reports()
    assert result["report"] == saved
    assert format_report(saved) != markdown
    validate_report_outputs(result, saved, selections, markdown)


@pytest.mark.parametrize("corruption", ["metric", "selection", "markdown"])
def test_validation_correction_still_rejects_changed_content(corruption):
    result, saved, selections, markdown = reports()
    if corruption == "metric":
        saved["diagnostics"]["numeric25"]["splits"]["validation"]["mean_seed"]["state"]["nll"] += .1
    elif corruption == "selection":
        selections["a"]["threshold"] += .1
    else:
        markdown += "Changed after sealing.\n"
    with pytest.raises(ValueError):
        validate_report_outputs(result, saved, selections, markdown)
