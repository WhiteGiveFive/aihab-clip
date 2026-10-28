"""A report invocation must authenticate the completed analysis payloads."""

import json

import pytest

from tools.analyze_attention_fusion_study import authenticated_analysis
from tools.prepare_attention_fusion_protocol import content_hash, file_hash


def seal(directory, files):
    manifest = {"status": "complete", "files": files}
    manifest["manifest_payload_sha256"] = content_hash(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_report_input_authentication_rejects_changed_payload(tmp_path):
    payload = tmp_path / "analysis.json"
    payload.write_text('{"study_id":"fixture"}')
    seal(tmp_path, {"analysis.json": file_hash(payload)})
    assert authenticated_analysis(tmp_path) == {"study_id": "fixture"}
    payload.write_text('{"study_id":"altered"}')
    with pytest.raises(ValueError, match="payload"):
        authenticated_analysis(tmp_path)


def test_report_input_authentication_rejects_manifest_tampering(tmp_path):
    payload = tmp_path / "analysis.json"
    payload.write_text("{}")
    seal(tmp_path, {"analysis.json": file_hash(payload)})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["status"] = "partial"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        authenticated_analysis(tmp_path)


def test_report_input_authentication_rejects_path_escape(tmp_path):
    directory = tmp_path / "analysis"
    directory.mkdir()
    payload = directory / "analysis.json"
    payload.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    seal(directory, {"analysis.json": file_hash(payload), "../outside.json": file_hash(outside)})
    with pytest.raises(ValueError, match="payload"):
        authenticated_analysis(directory)
