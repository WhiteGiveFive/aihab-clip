"""Additive validation correction for the sealed joint TCP router v1 study.

The original scorer renders insertion-ordered results, then writes a JSON copy
with sorted mapping keys. Regenerate Markdown from the recomputed report, whose
order matches scoring, after checking that its numeric content equals saved
JSON. All scientific sources, declaration and prior artifacts stay unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_joint_tcp as study
from multimodal.geo_helpfulness_joint_tcp_report import format_report, make_report


CORRECTION_ID = "joint_tcp_router_v1.validation_markdown_order.v1"


def validate_report_outputs(result: dict, saved_report: dict,
                            saved_selections: dict, saved_markdown: str) -> None:
    """Check values independently of mapping order, then exact original rendering."""
    if result["report"] != saved_report or result["selections"] != saved_selections:
        raise ValueError("Scored metrics or threshold selections do not reproduce")
    if format_report(result["report"]) != saved_markdown:
        raise ValueError("Markdown report differs")


def validate(output_dir=study.OUTPUT) -> dict:
    """Run every original verification with corrected Markdown input ordering."""
    root = study._output(output_dir)
    config = study.validate_declaration(root, check_parents=True)["config"]
    rows, parents, _ = study._scored_rows(root)
    folder = root / "scores"
    manifest = study._validate_stage(folder, root, role="joint_tcp_scores")
    if manifest["metadata"]["prediction_manifests"] != parents:
        raise ValueError("Scoring parents changed")
    pd.testing.assert_frame_equal(pd.read_parquet(folder / "scored_rows.parquet"), rows)
    result = make_report(rows, config)
    saved_report = study._json(folder / "report.json")
    result["report"]["all_predictions_verified_before_label_join_at"] = (
        saved_report["all_predictions_verified_before_label_join_at"]
    )
    result["report"]["rf_context"] = study._rf_context()
    validate_report_outputs(result, saved_report, study._json(folder / "selections.json"),
                            (folder / "report.md").read_text())
    if result["bootstrap_policy_order"] != study._json(folder / "bootstrap_policy_order.json"):
        raise ValueError("Bootstrap policy order differs")
    saved_draws = study._load_arrays(folder / "bootstrap_draws.npz")["draws"]
    if not np.array_equal(result["bootstrap_draws"], saved_draws):
        raise ValueError("Whole-plot bootstrap does not reproduce")
    pd.testing.assert_frame_equal(result["actions"], pd.read_parquet(folder / "actions.parquet"))
    return {"status": "valid", "fits": 64, "threshold_policies": 144,
            "locked_test_access": False, "validation_correction": CORRECTION_ID}
