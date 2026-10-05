"""Historical seed integer widths may change; predictions and event identity may not."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multimodal import geo_helpfulness_joint_tcp_capacity_test_score_compat as compat
from multimodal.geo_helpfulness_joint_tcp_capacity_report import (
    JointTCPReportError, make_test_report, policy_actions,
)
from test_multimodal_geo_joint_tcp_capacity_report import fixture, historical


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint64])
def test_seed_width_conversion_preserves_all_other_bytes_and_source(dtype):
    frame = pd.DataFrame({'row_uid': ['a', 'b', 'a', 'b'],
                          'training_seed': np.array([1, 2, 3, 4], dtype=dtype),
                          'raw_pred': np.array([2, 0, 1, 0], np.int8),
                          'geo_pred': np.array([0, 1, 2, 1], np.int64),
                          'acted': [True, False, True, False]})
    original = frame.copy(deep=True)
    metadata = {'record': {'source_sha256': 'a' * 64}}
    comparator = {'actions': frame, 'metadata': metadata}
    converted = compat.canonicalize_comparators(comparator)
    assert converted is not comparator and converted['actions'] is not frame
    assert converted['metadata'] is metadata
    assert converted['actions'].training_seed.dtype == np.int64
    np.testing.assert_array_equal(converted['actions'].training_seed, frame.training_seed)
    pd.testing.assert_frame_equal(converted['actions'].drop(columns='training_seed'), frame.drop(columns='training_seed'), check_exact=True)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)


@pytest.mark.parametrize('values', [
    [1., 2., 3., 4.], [1., 2.5, 3., 4.], [True, True, False, True],
    ['1', '2', '3', '4'], [1, 2, 3, 5], [0, 1, 2, 3], [1, 2, 3, None], [],
])
def test_bad_seed_metadata_is_rejected_before_cast(values):
    comparator = {'actions': pd.DataFrame({'training_seed': values}), 'metadata': {}}
    with pytest.raises(ValueError, match='integral values'):
        compat.canonicalize_comparators(comparator)


def test_actual_int8_archive_contract_reproduces_original_scorer_exactly():
    rows, config, selections = fixture()
    expected = historical(rows)
    archived = deepcopy(expected)
    archived['actions']['training_seed'] = archived['actions'].training_seed.astype(np.int8)
    blind = rows.drop(columns='label_id_dense')
    with pytest.raises(JointTCPReportError, match='identities differ'):
        policy_actions(blind, config, selections, archived)
    converted = compat.canonicalize_comparators(archived)
    pd.testing.assert_frame_equal(converted['actions'], expected['actions'], check_exact=True)
    actual = make_test_report(rows, config, selections, converted)
    reference = make_test_report(rows, config, selections, expected)
    assert actual['report'] == reference['report']
    pd.testing.assert_frame_equal(actual['actions'], reference['actions'], check_exact=True)
    np.testing.assert_array_equal(actual['bootstrap_draws'], reference['bootstrap_draws'])
    assert archived['actions'].training_seed.dtype == np.int8


def test_original_scoring_rows_are_returned_unchanged(monkeypatch):
    rows, _, _ = fixture()
    old = historical(rows)
    old['actions']['training_seed'] = old['actions'].training_seed.astype(np.int8)
    labels, declaration = object(), object()
    def source(actual_labels, actual_declaration):
        assert actual_labels is labels and actual_declaration is declaration
        return rows, old
    monkeypatch.setattr(compat.original, '_scoring_inputs', source)
    actual, comparator = compat._scoring_inputs(labels, declaration)
    assert actual is rows
    assert comparator['actions'].training_seed.dtype == np.int64


def setup_execution(tmp_path, monkeypatch):
    root, output = tmp_path / 'repo', tmp_path / 'output'
    root.mkdir()
    output.mkdir()
    (root / 'tools').mkdir()
    runner = root / 'tools/run_multimodal_geo_helpfulness_joint_tcp_capacity_v1.py'
    runner.write_text('# frozen corrected runner\n')
    monkeypatch.setattr(compat, 'ROOT', root)
    monkeypatch.setattr(compat, 'OUTPUT', output)
    declaration = {'study_id': compat.STUDY_ID, 'predictions': 'unchanged'}
    compat._write_json(output / 'declaration.json', declaration)
    digest = compat._hash(output / 'declaration.json')
    build = {'declaration_file_sha256': digest, 'prediction_identity': 'fixed'}
    reservation = {**build, 'status': 'reserved'}
    return declaration, build, reservation, runner


def test_execution_binds_and_reuses_same_event_without_rewriting(tmp_path, monkeypatch):
    declaration, build, reservation, runner = setup_execution(tmp_path, monkeypatch)
    record = compat.scoring_execution(declaration, build, reservation, create=True)
    path = compat.OUTPUT / 'scoring_execution.json'
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert compat.scoring_execution(declaration, build, reservation) == record
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert record['reserved_build_sha256'] == compat.canonical_sha256(build)
    assert record['reservation_sha256'] == compat.canonical_sha256(reservation)
    assert record['code_file_sha256'][str(runner)] == compat._hash(runner)
    assert record['code_file_sha256'][str(Path(compat.__file__).resolve())] == compat._hash(Path(compat.__file__))
    assert record['scientific_changes'] is False
    assert record['new_evaluation_event'] is False


@pytest.mark.parametrize('corruption', ['runner', 'module', 'build', 'reservation', 'declaration', 'missing'])
def test_execution_rejects_code_or_event_drift(tmp_path, monkeypatch, corruption):
    declaration, build, reservation, runner = setup_execution(tmp_path, monkeypatch)
    compat.scoring_execution(declaration, build, reservation, create=True)
    if corruption == 'runner':
        runner.write_text('# altered\n')
    elif corruption == 'module':
        actual = compat._hash
        monkeypatch.setattr(compat, '_hash', lambda path: 'f' * 64 if Path(path).resolve() == Path(compat.__file__).resolve() else actual(path))
    elif corruption == 'build':
        build = {**build, 'prediction_identity': 'changed'}
    elif corruption == 'reservation':
        reservation = {**reservation, 'status': 'other'}
    elif corruption == 'declaration':
        declaration = {**declaration, 'predictions': 'changed'}
    else:
        (compat.OUTPUT / 'scoring_execution.json').unlink()
    with pytest.raises(ValueError):
        compat.scoring_execution(declaration, build, reservation)
