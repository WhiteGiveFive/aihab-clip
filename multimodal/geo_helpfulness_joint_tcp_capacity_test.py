"""Frozen depth-study routers applied to the September 15 legacy expert family."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from multimodal import geo_helpfulness_joint_tcp_capacity as study
from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage,
    _staging, _validate_stage, _write_json,
)
from multimodal.geo_helpfulness_protocol import canonical_sha256
from multimodal.geo_helpfulness_joint_tcp_test import _check_snapshot
from multimodal.geo_helpfulness_joint_tcp_capacity_test_inputs import (
    authenticate_sources, prepare_seed, load_historical_actions,
)

ROOT, BASE, SEEDS = study.ROOT, study.BASE, study.SEEDS
OUTPUT = BASE / 'joint_tcp_router_capacity_legacy_test_v1'
STUDY_ID = 'joint_tcp_router_capacity_legacy_test_v1'
FUSION_REFERENCE = {
    'accuracy': 0.7431328878990349,
    'weighted_f1': 0.7383523180768167,
}


def declare():
    with _lock(OUTPUT, 'declare'):
        if (OUTPUT / 'declaration.json').exists():
            validate_declaration(deep=True)
            return {'status': 'reused_valid'}
        parent = study.validate_declaration(deep=True)
        _validate_stage(study.OUTPUT / 'scores', study.OUTPUT, role='joint_capacity_scores')
        # Authenticate all fitted predictions before binding the test application.
        study.all_predictions()
        context = authenticate_sources()
        previous_root = BASE / 'joint_tcp_router_test_v1'
        previous = _json(previous_root / 'declaration.json')
        label_source = previous_root / 'label_acquisition/labels.parquet'
        receipt = _json(previous_root / 'scores/receipt.json')
        _validate_stage(previous_root / 'scores', previous_root, role='joint_tcp_cleaned_test_scores')
        if _hash(label_source) != receipt['labels_file_sha256']:
            raise ValueError('Previously acquired canonical label source changed')
        registry_root = previous['registry_root']
        if context['identity_manifest'] != previous['identity_manifest']:
            raise ValueError('Historical and current cleaned-test identities differ')
        _write_json(OUTPUT / 'declaration.json', {
            'schema_version': 'geo_helpfulness.joint_tcp_capacity_test_declaration.v1',
            'study_id': STUDY_ID, 'created_at': _now(), 'config': parent['config'],
            'user_authorization': 'Apply deeper joint routers and controls to exact September15 fixed-final experts on cleaned test',
            'parent_snapshot': study.snapshot(study.OUTPUT), 'original_joint_snapshot': study.snapshot(study.PARENT),
            'source_file_sha256': parent['source_file_sha256'],
            'legacy_context': context,
            'validation_selections': _json(study.OUTPUT / 'scores/selections.json'),
            'identity_manifest': context['identity_manifest'], 'ontology': context['ontology'],
            'label_source': {'path': str(label_source), 'file_sha256': _hash(label_source)},
            'registry_root': registry_root, 'prior_registry_snapshot': study.snapshot(Path(registry_root)),
            'test_threshold_selection': False, 'router_fitting': False,
            'adaptive_reuse': True, 'independent_confirmation': False,
        })
    return {'status': 'declared', 'routers': 160, 'expert_family': 'best_model'}


def validate_declaration(*, deep=False):
    saved = _json(OUTPUT / 'declaration.json')
    if saved['study_id'] != STUDY_ID or saved['config'] != study.configuration():
        raise ValueError('Test application scope changed')
    study.validate_declaration(deep=deep)
    _check_snapshot(study.OUTPUT, saved['parent_snapshot'])
    _check_snapshot(study.PARENT, saved['original_joint_snapshot'])
    for path, digest in saved['legacy_context']['source_file_sha256'].items():
        if _hash(Path(path)) != digest:
            raise ValueError('Legacy sources changed')
    if saved['validation_selections'] != _json(study.OUTPUT / 'scores/selections.json'):
        raise ValueError('Frozen validation thresholds changed')
    if deep:
        _check_snapshot(Path(saved['registry_root']), saved['prior_registry_snapshot'], allow_additions=True)
    return saved


def _infer(seed, features, config):
    frames, diagnostics = [], {}
    for spec in config['configs']:
        state = study.model_state(seed, spec)
        matrix = features[f"X_{spec['input_id']}"]
        native = study.predict(state, matrix)
        repeated = study.predict(state, matrix)
        portable = study.predict(state, matrix, native=False)
        if any(not np.array_equal(native[k], repeated[k]) for k in study.PREDICTION_KEYS):
            raise ValueError('Test native inference does not reproduce exactly')
        diagnostics[spec['config_id']] = {key: {
            'max_abs_error': float(np.abs(native[key].astype(np.float64) - portable[key]).max()),
            'mean_abs_error': float(np.abs(native[key].astype(np.float64) - portable[key]).mean()),
        } for key in study.PREDICTION_KEYS}
        diagnostics[spec['config_id']]['native_float32_exact'] = True
        frames.append(study.prediction_frame(features, native, seed, spec, 'test'))
    return pd.concat(frames, ignore_index=True), diagnostics


def predict_seed(seed):
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import policy_actions
    declaration = validate_declaration()
    with _lock(OUTPUT, f'predict_{seed}'):
        final = OUTPUT / f'predictions/seed_{seed}'
        if final.exists():
            predictions(seed, replay=True)
            return {'status': 'reused_valid', 'seed': seed}
        prepared = prepare_seed(seed, context=declaration['legacy_context'])
        features = prepared['test']
        rows, diagnostics = _infer(seed, features, declaration['config'])
        actions = policy_actions(rows, declaration['config'], declaration['validation_selections'])
        if any('label' in key or key == 'target_state' for key in (*rows.columns, *actions.columns)):
            raise ValueError('Label-free prediction stage contains labels')
        staged = _staging(OUTPUT, f'predictions_{seed}')
        rows.to_parquet(staged / 'rows.parquet', index=False)
        actions.to_parquet(staged / 'actions.parquet', index=False)
        _save_arrays(staged / 'features.npz', features)
        _write_json(staged / 'source_replay.json', prepared['metadata'])
        _write_json(staged / 'float64_replay.json', diagnostics)
        _seal_stage(staged, final, OUTPUT, role='joint_tcp_cleaned_test_predictions', metadata={'seed': seed})
        predictions(seed, replay=True)
    return {'status': 'complete', 'seed': seed, 'routers': 40}


def predictions(seed, *, replay=False):
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import policy_actions
    folder = OUTPUT / f'predictions/seed_{seed}'
    seal = _validate_stage(folder, OUTPUT, role='joint_tcp_cleaned_test_predictions')
    if seal['metadata'] != {'seed': seed}:
        raise ValueError('Test prediction seed mismatch')
    rows, actions = [pd.read_parquet(folder / name) for name in ('rows.parquet', 'actions.parquet')]
    if any('label' in key or key == 'target_state' for key in (*rows.columns, *actions.columns)):
        raise ValueError('Label-free predictions contain labels')
    if replay:
        declaration = _json(OUTPUT / 'declaration.json')
        features = _load_arrays(folder / 'features.npz')
        prepared = prepare_seed(seed, context=declaration['legacy_context'])
        if set(features) != set(prepared['test']):
            raise ValueError('Sealed test feature payload differs from archived sources')
        for name, expected_feature in prepared['test'].items():
            if (features[name].dtype != expected_feature.dtype
                    or not np.array_equal(features[name], expected_feature)):
                raise ValueError(f'Sealed test feature replay differs: {name}')
        if prepared['metadata'] != _json(folder / 'source_replay.json'):
            raise ValueError('Archived expert source replay metadata changed')
        native, diagnostics = _infer(seed, features, declaration['config'])
        expected = policy_actions(native, declaration['config'], declaration['validation_selections'])
        pd.testing.assert_frame_equal(native, rows, check_exact=True)
        pd.testing.assert_frame_equal(expected, actions, check_exact=True)
        if diagnostics != _json(folder / 'float64_replay.json'):
            raise ValueError('Float64 replay diagnostic changed')
    return rows, actions


def _scoring_inputs(labels, declaration):
    rows = pd.concat([predictions(seed)[0] for seed in SEEDS], ignore_index=True)
    scored = rows.merge(labels[['row_uid', 'plot_idx', 'label_id_dense']], on=['row_uid', 'plot_idx'], how='left', sort=False, validate='many_to_one')
    if len(scored) != len(rows) or scored.label_id_dense.isna().any():
        raise ValueError('Canonical test label join failed')
    scored['label_id_dense'] = scored.label_id_dense.astype(np.int64)
    _assert_fusion_reference(scored)
    comparators = load_historical_actions(context=declaration['legacy_context'])
    return scored, comparators


def _classification_reference(labels, predicted):
    """Independent accuracy/F1 check, separate from report metric helpers."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    if (labels.ndim != 1 or labels.shape != predicted.shape or not len(labels)
            or labels.dtype.kind not in 'iu' or predicted.dtype.kind not in 'iu'
            or np.any(labels < 0) or np.any(labels >= 18)
            or np.any(predicted < 0) or np.any(predicted >= 18)):
        raise ValueError('Invalid reference classification labels or predictions')
    cm = np.bincount(18 * labels + predicted, minlength=18 * 18).reshape(18, 18)
    support, predicted_count = cm.sum(axis=1), cm.sum(axis=0)
    denominator = support + predicted_count
    f1 = np.divide(2 * np.diag(cm), denominator,
                   out=np.zeros(18, dtype=np.float64), where=denominator > 0)
    return {'accuracy': float(np.trace(cm) / cm.sum()),
            'weighted_f1': float(np.dot(f1, support) / cm.sum())}


def _assert_fusion_reference(rows):
    """After label joining, require the exact report's expert baseline."""
    observed = []
    if set(rows.training_seed) != set(SEEDS):
        raise ValueError('Historical fusion reference requires all four seeds')
    for seed in SEEDS:
        part = rows.loc[rows.training_seed.eq(seed)]
        first_config = part.config_id.iloc[0]
        part = part.loc[part.config_id.eq(first_config)].sort_values('row_uid')
        if len(part) != 1347 or part.row_uid.duplicated().any() or part.plot_idx.nunique() != 531:
            raise ValueError('Historical fusion reference identity count differs')
        predicted = np.stack(part.raw_logits).argmax(axis=1)
        observed.append(_classification_reference(part.label_id_dense.to_numpy(), predicted))
    mean = {name: float(np.mean([value[name] for value in observed])) for name in FUSION_REFERENCE}
    if any(not np.isclose(mean[name], expected, atol=1e-12, rtol=0)
           for name, expected in FUSION_REFERENCE.items()):
        raise ValueError(f'September15 fusion baseline was not reproduced: {mean}')
    return {'per_seed': {str(seed): value for seed, value in zip(SEEDS, observed)}, 'mean': mean}


def _verify_saved_actions(result):
    saved = pd.concat([predictions(seed)[1] for seed in SEEDS], ignore_index=True).sort_values(['row_uid', 'training_seed']).reset_index(drop=True)
    actual = result['actions'].sort_values(['row_uid', 'training_seed']).reset_index(drop=True)
    for name in saved:
        if name not in actual or not np.array_equal(saved[name].to_numpy(), actual[name].to_numpy()):
            raise ValueError(f'Scoring changed label-free actions: {name}')


def score():
    from multimodal import geo_helpfulness_joint_tcp_capacity_test_registry as registry
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import make_test_report
    declaration = validate_declaration(deep=True)
    with _lock(OUTPUT, 'score'):
        final = OUTPUT / 'scores'
        if final.exists():
            _validate_stage(final, OUTPUT, role='joint_capacity_legacy_test_scores')
            registry.finish_event(declaration['registry_root'], _json(final / 'build.json'),
                _json(final / 'reservation.json'), _json(final / 'receipt.json'), final / 'receipt.json')
            validate()
            return {'status': 'reused_valid'}
        for seed in SEEDS:
            predictions(seed, replay=True)
        manifests = {seed: OUTPUT / f'predictions/seed_{seed}/manifest.json' for seed in SEEDS}
        build = registry.make_build_record(OUTPUT, declaration, manifests)
        reservation = registry.reserve_event(declaration['registry_root'], build)
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
        receipt = {'study_id': STUDY_ID, 'created_at': _now(), 'evaluation_event_id': canonical_sha256(build), **build,
                   'report_file_sha256': _hash(staged / 'report.json'),
                   'labels_file_sha256': _hash(acquisition / 'labels.parquet'),
                   'label_source_receipt_sha256': _hash(acquisition / 'source_receipt.json'),
                   'adaptive_reuse': True, 'no_fitting_or_selection': True}
        receipt['receipt_sha256'] = canonical_sha256(receipt)
        _write_json(staged / 'receipt.json', receipt)
        _seal_stage(staged, final, OUTPUT, role='joint_capacity_legacy_test_scores', metadata={
            'prediction_manifests': {str(seed): _hash(path) for seed, path in manifests.items()}})
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
