"""Additive integer-metadata compatibility for already frozen test scoring.

The historical archive uses int8 training seeds while newly generated router
rows use int64. The frozen scorer correctly requires the same identity values,
but its DataFrame.equals check additionally requires identical dtype widths.
Only that metadata representation is canonicalized. Scientific source modules,
models, predictions, thresholds, labels and existing registry ownership remain
unchanged. A separate execution record authenticates this adapter on every use.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_joint_tcp_capacity_test as original
from multimodal.geo_helpfulness_joint_tcp_capacity_test import (
    ROOT, OUTPUT, STUDY_ID, SEEDS, study, predictions, validate_declaration,
    _verify_saved_actions,
)
from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _seal_stage, _staging, _validate_stage, _write_json,
)
from multimodal.geo_helpfulness_protocol import canonical_sha256

def canonicalize_comparators(comparators):
    """Copy comparator seed metadata to int64 without changing any value."""
    if not isinstance(comparators, dict) or set(comparators) != {'actions', 'metadata'}:
        raise ValueError('Historical comparators require actions and metadata')
    frame = comparators['actions']
    if not isinstance(frame, pd.DataFrame) or 'training_seed' not in frame:
        raise ValueError('Historical comparator seed metadata is missing')
    seeds = frame['training_seed']
    if (seeds.dtype.kind not in 'iu' or seeds.isna().any()
            or seeds.empty or not seeds.isin(SEEDS).all()):
        raise ValueError('Historical training seeds must be integral values 1 through 4')
    converted = frame.copy(deep=True)
    converted['training_seed'] = seeds.astype(np.int64)
    if not np.array_equal(seeds.to_numpy(), converted['training_seed'].to_numpy()):
        raise ValueError('Integer metadata conversion changed seed values')
    pd.testing.assert_frame_equal(frame.drop(columns='training_seed'),
                                  converted.drop(columns='training_seed'), check_exact=True)
    return {**comparators, 'actions': converted}


def _scoring_inputs(labels, declaration):
    rows, comparators = original._scoring_inputs(labels, declaration)
    return rows, canonicalize_comparators(comparators)


def _execution_fields(declaration, build, reservation):
    source = Path(__file__).resolve()
    runner = ROOT / 'tools/run_multimodal_geo_helpfulness_joint_tcp_capacity_v1.py'
    code = {str(source): _hash(source)}
    if runner.exists():
        code[str(runner)] = _hash(runner)
    declaration_hash = _hash(OUTPUT / 'declaration.json')
    if (_json(OUTPUT / 'declaration.json') != declaration
            or build['declaration_file_sha256'] != declaration_hash
            or reservation['declaration_file_sha256'] != declaration_hash):
        raise ValueError('Scoring compatibility may not change the reserved declaration')
    return {
        'schema_version': 'geo_helpfulness.joint_tcp_capacity_scoring_compatibility.v1',
        'study_id': STUDY_ID,
        'reason': 'Historical training_seed int8 and native prediction training_seed int64 contain identical identities; DataFrame.equals requires equal dtype widths.',
        'only_adaptation': 'copy historical action table and cast integer training_seed metadata to int64 with exact value preservation',
        'scientific_changes': False, 'prediction_or_action_changes': False,
        'model_or_threshold_changes': False, 'new_evaluation_event': False,
        'original_declaration_file_sha256': declaration_hash,
        'reserved_build_sha256': canonical_sha256(build),
        'reservation_sha256': canonical_sha256(reservation),
        'evaluation_event_id': canonical_sha256(build),
        'code_file_sha256': code,
    }


def scoring_execution(declaration, build, reservation, *, create=False):
    """Bind the additive execution to the original, already reserved bundle."""
    expected = _execution_fields(declaration, build, reservation)
    path = OUTPUT / 'scoring_execution.json'
    if path.exists():
        record = _json(path)
        if ({key: value for key, value in record.items() if key != 'created_at'} != expected
                or not isinstance(record.get('created_at'), str)):
            raise ValueError('Scoring execution code or original event binding changed')
        return record
    if not create:
        raise ValueError('Scoring compatibility execution record is missing')
    record = {**expected, 'created_at': _now()}
    _write_json(path, record)
    return record


def _validate_execution_receipt(final, declaration):
    build, reservation = _json(final / 'build.json'), _json(final / 'reservation.json')
    expected = scoring_execution(declaration, build, reservation)
    receipt = _json(final / 'receipt.json')
    if (_json(final / 'scoring_execution.json') != expected
            or _hash(final / 'scoring_execution.json') != _hash(OUTPUT / 'scoring_execution.json')
            or receipt.get('scoring_execution_file_sha256') != _hash(final / 'scoring_execution.json')
            or receipt.get('scoring_execution_sha256') != canonical_sha256(expected)
            or receipt.get('scoring_execution_code_sha256') != expected['code_file_sha256']):
        raise ValueError('Scoring receipt does not authenticate compatibility execution')
    stage = _validate_stage(final, OUTPUT, role='joint_capacity_legacy_test_scores')
    if stage['metadata'].get('scoring_execution_sha256') != canonical_sha256(expected):
        raise ValueError('Score manifest does not bind scoring compatibility execution')


def score():
    from multimodal import geo_helpfulness_joint_tcp_capacity_test_registry as registry
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import make_test_report
    declaration = validate_declaration(deep=True)
    with _lock(OUTPUT, 'score'):
        final = OUTPUT / 'scores'
        if final.exists():
            _validate_stage(final, OUTPUT, role='joint_capacity_legacy_test_scores')
            _validate_execution_receipt(final, declaration)
            registry.finish_event(declaration['registry_root'], _json(final / 'build.json'),
                _json(final / 'reservation.json'), _json(final / 'receipt.json'), final / 'receipt.json')
            validate()
            return {'status': 'reused_valid'}
        for seed in SEEDS:
            predictions(seed, replay=True)
        manifests = {seed: OUTPUT / f'predictions/seed_{seed}/manifest.json' for seed in SEEDS}
        build = registry.make_build_record(OUTPUT, declaration, manifests)
        reservation = registry.reserve_event(declaration['registry_root'], build)
        execution = scoring_execution(declaration, build, reservation, create=True)
        acquisition = OUTPUT / 'label_acquisition'
        acquisition.mkdir(exist_ok=True)
        labels, _ = registry.acquire_labels(declaration['registry_root'], build, reservation, declaration, acquisition)
        rows, comparators = _scoring_inputs(labels, declaration)
        result = make_test_report(rows, declaration['config'], declaration['validation_selections'], comparators=comparators)
        _verify_saved_actions(result)
        staged = _staging(OUTPUT, 'scores')
        study.save_report(staged, result, rows)
        _write_json(staged / 'build.json', build)
        _write_json(staged / 'reservation.json', reservation)
        _write_json(staged / 'scoring_execution.json', execution)
        receipt = {'study_id': STUDY_ID, 'created_at': _now(), 'evaluation_event_id': canonical_sha256(build), **build,
                   'report_file_sha256': _hash(staged / 'report.json'),
                   'labels_file_sha256': _hash(acquisition / 'labels.parquet'),
                   'label_source_receipt_sha256': _hash(acquisition / 'source_receipt.json'),
                   'adaptive_reuse': True, 'no_fitting_or_selection': True,
                   'scoring_execution_file_sha256': _hash(staged / 'scoring_execution.json'),
                   'scoring_execution_sha256': canonical_sha256(execution),
                   'scoring_execution_code_sha256': execution['code_file_sha256']}
        receipt['receipt_sha256'] = canonical_sha256(receipt)
        _write_json(staged / 'receipt.json', receipt)
        _seal_stage(staged, final, OUTPUT, role='joint_capacity_legacy_test_scores', metadata={
            'prediction_manifests': {str(seed): _hash(path) for seed, path in manifests.items()},
            'scoring_execution_sha256': canonical_sha256(execution)})
        registry.finish_event(declaration['registry_root'], build, reservation, receipt, final / 'receipt.json')
    return {'status': 'complete', 'primary_policies': 401, 'historical_policies': 6, 'contrasts': 1100}


def validate():
    from multimodal import geo_helpfulness_joint_tcp_capacity_test_registry as registry
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import make_test_report, format_report
    declaration = validate_declaration(deep=True)
    for seed in SEEDS:
        predictions(seed, replay=True)
    final = OUTPUT / 'scores'
    _validate_stage(final, OUTPUT, role='joint_capacity_legacy_test_scores')
    manifests = {seed: OUTPUT / f'predictions/seed_{seed}/manifest.json' for seed in SEEDS}
    build = registry.make_build_record(OUTPUT, declaration, manifests)
    if build != _json(final / 'build.json'):
        raise ValueError('Test score identity changed')
    receipt, reservation = _json(final / 'receipt.json'), _json(final / 'reservation.json')
    from multimodal.geo_helpfulness_router_m4_common import check_self_hash
    check_self_hash(receipt, 'receipt_sha256')
    if (receipt.get('study_id') != STUDY_ID
            or receipt['report_file_sha256'] != _hash(final / 'report.json')
            or receipt['labels_file_sha256'] != _hash(OUTPUT / 'label_acquisition/labels.parquet')
            or receipt['label_source_receipt_sha256'] != _hash(OUTPUT / 'label_acquisition/source_receipt.json')):
        raise ValueError('Test scoring receipt does not bind report, labels and source receipt')
    _validate_execution_receipt(final, declaration)
    registry.validate_event(declaration['registry_root'], build, reservation, receipt, final / 'receipt.json')
    labels, _ = registry.acquire_labels(declaration['registry_root'], build, reservation, declaration, OUTPUT / 'label_acquisition')
    rows, comparators = _scoring_inputs(labels, declaration)
    result = make_test_report(rows, declaration['config'], declaration['validation_selections'], comparators=comparators)
    _verify_saved_actions(result)
    if result['report'] != _json(final / 'report.json'):
        raise ValueError('Legacy test metrics do not reproduce')
    if format_report(result['report']) != (final / 'report.md').read_text():
        raise ValueError('Legacy test markdown does not reproduce')
    pd.testing.assert_frame_equal(rows, pd.read_parquet(final / 'scored_rows.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(result['actions'], pd.read_parquet(final / 'actions.parquet'), check_exact=True)
    if result['bootstrap_policy_order'] != _json(final / 'bootstrap_policy_order.json') or not np.array_equal(result['bootstrap_draws'], _load_arrays(final / 'bootstrap_draws.npz')['draws']):
        raise ValueError('Legacy test bootstrap does not reproduce')
    return {'status': 'valid', 'routers': 160, 'primary_policies': 401, 'historical_policies': 6, 'contrasts': 1100}
