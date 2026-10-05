"""Additive, sealed depth study for the four-state router and auxiliary TCP loss."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage,
    _staging, _validate_stage, _write_json,
)
from multimodal import geo_helpfulness_joint_tcp as parent
from multimodal.geo_helpfulness_router_candidate_state import TARGET_ORDER, variant_feature_indices

ROOT, BASE, PROTOCOL = parent.ROOT, parent.BASE, parent.PROTOCOL
PARENT = parent.OUTPUT
OUTPUT = BASE / 'joint_tcp_router_capacity_v1'
CONFIG = ROOT / 'configs/multimodal_geo_joint_tcp_capacity_v1.json'
PLAN = ROOT / 'dev_plans/reliability_aware_geo_fusion/20_joint_tcp_router_capacity.md'
SEEDS, SPLITS, PREDICTION_KEYS = parent.SEEDS, parent.SPLITS, parent.PREDICTION_KEYS
ARCHITECTURES = ((32,), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32))
CODE_PATHS = (
    'multimodal/geo_helpfulness_joint_tcp_capacity.py',
    'multimodal/geo_helpfulness_joint_tcp_capacity_fit.py',
    'multimodal/geo_helpfulness_joint_tcp_capacity_report.py',
    'multimodal/geo_helpfulness_joint_tcp_capacity_test.py',
    'multimodal/geo_helpfulness_joint_tcp_capacity_test_inputs.py',
    'multimodal/geo_helpfulness_joint_tcp_capacity_test_registry.py',
    'tools/run_multimodal_geo_helpfulness_joint_tcp_capacity.py',
)


def configuration():
    """Reject any drift from the approved grid, recipe and evaluation contract."""
    from multimodal.geo_helpfulness_joint_tcp_fit import DEFAULT_PARAMETERS
    from multimodal.geo_helpfulness_joint_tcp_capacity_fit import parameter_count
    from multimodal.geo_helpfulness_protocol import canonical_sha256

    cfg = _json(CONFIG)
    expected = {'approval_status': 'approved_user_selected_moderate_depth_grid_2026_09_28',
     'architecture_selection': 'none_report_all',
     'auxiliary_weights': [0.0, 0.1, 1.0, 10.0],
     'bootstrap': {'resamples': 2000,
                   'seed': 20260826,
                   'selection_adjusted': False,
                   'unit': 'whole_plot_all_images_all_seed_realizations'},
     'configuration_count': 40,
     'evidence_status': 'exploratory_reused_development_validation_after_prior_research_test_exposure',
     'expert_updates': False,
     'fixed_zero_policy_count': 40,
     'harm_denominator': 10,
     'harm_numerator_grid': [5, 7, 9],
     'hidden_layer_rule': 'linear_relu_dropout_each_layer',
     'locked_test_access': True,
     'loss': 'mean_image_cross_entropy_plus_lambda_times_mean_image_and_expert_tcp_mse',
     'minimum_coverage_fraction': 0.01,
     'minimum_plots_grid': [20, 10, 0],
     'new_extraction': False,
     'new_fit_count': 128,
     'paired_contrasts': {'development_total': 1020,
                          'larger_architecture_versus_32': 320,
                          'positive_auxiliary_weight_versus_zero': 300,
                          'test_matched_RF_LR_at_plots20_harm0p7': 80,
                          'test_total': 1100,
                          'versus_fusion': 400},
     'parent_study_id': 'joint_tcp_router_v1',
     'positive_net_required': True,
     'reference_policy_count': 1,
     'relaxed_policy_count': 360,
     'reused_fit_count': 32,
     'schema_version': 'geo_helpfulness.joint_tcp_capacity_config.v1',
     'score_dtype': 'float64_after_native_float32_inference',
     'study_id': 'joint_tcp_router_capacity_v1',
     'target_order': ['rescue', 'harm', 'both_correct', 'both_wrong'],
     'tcp_order': ['fusion', 'geo'],
     'tcp_target': 'authoritative_native_t1_true_class_probability',
     'test_application': {'encoder_inputs': 'authenticated_historical_cached_image_features',
                          'encoder_provenance_limitation': 'No historical end-to-end '
                                                           'encoder-to-cache fingerprint; exact '
                                                           'caches, heads and saved outputs '
                                                           'authenticated.',
                          'expected_fusion_mean_accuracy': 0.7431328878990348,
                          'expert_family': 'legacy_fixed_final_epoch50_best_model',
                          'expert_fit_rows': 4200,
                          'historical_comparators': ['rf_full727',
                                                     'rf_no_pairs79',
                                                     'rf_numeric25',
                                                     'lr_full727',
                                                     'lr_no_pairs79',
                                                     'lr_numeric25'],
                          'router_refitting': False,
                          'test_plots': 531,
                          'test_rows': 1347,
                          'threshold_selection': False},
     'training_seeds': [1, 2, 3, 4],
     'verification': {'native_float32': 'exact_serialized_state_replay',
                      'portable_float64': 'diagnostic_only'},
     'weight_selection': 'none_report_all',
     'with_embeddings': False}
    expected['parameters'] = {k: v for k, v in DEFAULT_PARAMETERS.items() if k != 'hidden_width'}
    expected['features'] = [
        {'input_dim': 25, 'input_id': 'numeric25', 'variant': 'ablation_numeric25'},
        {'input_dim': 79, 'input_id': 'no_pairs79', 'variant': 'ablation_no_pairs79'},
    ]
    expected['architectures'], expected['configs'] = [], []
    for widths in ARCHITECTURES:
        architecture = 'h' + '_'.join(map(str, widths))
        source = 'reuse_parent' if widths == (32,) else 'new_fit'
        expected['architectures'].append({
            'architecture_id': architecture, 'hidden_widths': list(widths), 'source': source,
            'parameter_counts': {str(d): parameter_count(d, list(widths)) for d in (25, 79)},
        })
        for feature in expected['features']:
            for auxiliary_weight, token in ((0., '0'), (.1, '0p1'), (1., '1'), (10., '10')):
                prefix = 'joint' if source == 'reuse_parent' else f'joint_capacity__{architecture}'
                expected['configs'].append({
                    **feature, 'hidden_widths': list(widths), 'architecture_id': architecture,
                    'auxiliary_weight': auxiliary_weight, 'source': source, 'with_embeddings': False,
                    'config_id': f"{prefix}__{feature['input_id']}__lambda_{token}",
                })
    # Canonical JSON comparison distinguishes booleans from integers and retains
    # declared order while tolerating harmless JSON whitespace/key ordering.
    if canonical_sha256(cfg) != canonical_sha256(expected):
        raise ValueError('Capacity approved configuration, grid or recipe changed')
    return cfg


def snapshot(directory):
    return {str(p.relative_to(directory)): {'sha256': _hash(p), 'size': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns} for p in sorted(directory.rglob('*'))
            if p.is_file() and not p.name.endswith('.lock')}


def declare():
    with _lock(OUTPUT, 'declare'):
        if (OUTPUT / 'declaration.json').exists():
            validate_declaration(deep=True)
            return {'status': 'reused_valid'}
        cfg = configuration()
        saved = parent.validate_declaration(PARENT, check_parents=True)
        for path in sorted(PARENT.rglob('manifest.json')):
            _validate_stage(path.parent, PARENT, role=_json(path)['role'])
        from multimodal.geo_helpfulness_joint_tcp_capacity_test_inputs import authenticate_sources
        context = authenticate_sources()
        files = {ROOT / p for p in saved['file_sha256']}
        files.update(ROOT / p for p in CODE_PATHS)
        files.update((ROOT / 'multimodal').glob('*.py'))
        files.add(CONFIG)
        if not all(p.is_file() for p in files):
            raise ValueError('Capacity implementation is incomplete')
        _write_json(OUTPUT / 'declaration.json', {
            'schema_version': 'geo_helpfulness.joint_tcp_capacity_declaration.v1',
            'study_id': cfg['study_id'], 'config': cfg, 'created_at': _now(),
            'user_authorization': 'Four selected moderate two/three-layer MLPs and cleaned-test comparison using September15 experts',
            'approved_plan_text': PLAN.read_text(),
            'source_file_sha256': {str(p.relative_to(ROOT)): _hash(p) for p in sorted(files)},
            'parent_snapshot': snapshot(PARENT),
            'feature_order': {s['input_id']: parent.feature_names(s) for s in cfg['configs']},
            'environment': {name: version(name) for name in ('numpy', 'pandas', 'pyarrow', 'torch', 'scikit-learn')},
            'validation_labels': 'joined only after all128newmodels_and32controls_predictions_sealed',
            'legacy_source_audit': context,
        })
    return {'status': 'declared', 'new_fits': 128, 'reused_fits': 32}


def validate_declaration(*, deep=False):
    value = _json(OUTPUT / 'declaration.json')
    if value['config'] != configuration():
        raise ValueError('Declared capacity configuration changed')
    for path, digest in value['source_file_sha256'].items():
        if deep or path.endswith('.py') or ROOT / path == CONFIG:
            if _hash(ROOT / path) != digest:
                raise ValueError(f'Declared source changed: {path}')
    if deep and snapshot(PARENT) != value['parent_snapshot']:
        raise ValueError('Original joint study changed')
    if deep:
        for path, digest in value['legacy_source_audit']['source_file_sha256'].items():
            if _hash(Path(path)) != digest:
                raise ValueError(f'Legacy expert/cache source changed: {path}')
    return value


def prepare_seed(seed):
    if seed not in SEEDS:
        raise ValueError('Unknown seed')
    validate_declaration()
    with _lock(OUTPUT, f'prepare_{seed}'):
        folder = OUTPUT / f'inputs/seed_{seed}'
        if folder.exists():
            load_inputs(seed)
            return {'status': 'reused_valid', 'seed': seed}
        arrays = parent._features(PARENT, seed)
        data = parent._inputs(PARENT, seed, arrays)
        staged = _staging(OUTPUT, f'inputs_{seed}')
        saved = {'y_train': data['y_train'], 'tcp_targets': data['tcp_targets']}
        for split in SPLITS:
            for name in ('row_uid', 'plot_idx', 'producer_fold', 'image_logits', 'geo_logits', 'raw_logits'):
                saved[f'{split}_{name}'] = arrays[split][name]
            for short, variant in [('numeric25', 'ablation_numeric25'), ('no_pairs79', 'ablation_no_pairs79')]:
                saved[f'{split}_{short}'] = np.ascontiguousarray(data[f'X_{split}'][:, variant_feature_indices(variant)], dtype=np.float32)
        _save_arrays(staged / 'inputs.npz', saved)
        _write_json(staged / 'lineage.json', {
            'features_manifest_sha256': _hash(PARENT / f'features/seed_{seed}/manifest.json'),
            'router_inputs_manifest_sha256': _hash(PARENT / 'router_inputs/manifest.json'),
            'native_tcp_targets_verified': True, 'four_state_targets_verified': True,
            'expert_fit_prediction_plot_separation': 'authenticated_parent_producer_manifests',
            'validation_labels_present': False,
        })
        _seal_stage(staged, folder, OUTPUT, role='joint_capacity_inputs', metadata={'seed': seed})
    return {'status': 'complete', 'seed': seed}


def load_inputs(seed):
    folder = OUTPUT / f'inputs/seed_{seed}'
    _validate_stage(folder, OUTPUT, role='joint_capacity_inputs')
    data = _load_arrays(folder / 'inputs.npz')
    for split, rows, plots in [('train', 3378, 1300), ('validation', 822, 325)]:
        if len(data[f'{split}_row_uid']) != rows or len(np.unique(data[f'{split}_plot_idx'])) != plots:
            raise ValueError('Router input identity changed')
        for short, width in [('numeric25', 25), ('no_pairs79', 79)]:
            matrix = data[f'{split}_{short}']
            if matrix.shape != (rows, width) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                raise ValueError('Router input width/order/dtype changed')
    if set(data['train_plot_idx']) & set(data['validation_plot_idx']):
        raise ValueError('Development plot overlap')
    if any('validation_label' in k or 'validation_target' in k for k in data):
        raise ValueError('Validation labels entered preparation')
    return data


def _state_folder(seed, spec):
    root = PARENT if spec['source'] == 'reuse_parent' else OUTPUT
    return root, root / f"models/seed_{seed}/{spec['config_id']}"


def model_state(seed, spec):
    root, folder = _state_folder(seed, spec)
    _validate_stage(folder, root, role='joint_tcp_model' if root == PARENT else 'joint_capacity_model')
    state = _json(folder / 'state.json')
    expected = {'training_seed': seed, 'input_dim': spec['input_dim'], 'auxiliary_weight': spec['auxiliary_weight'],
                'config_id': spec['config_id'], 'fit_row_count': 3378}
    parameters = configuration()['parameters']
    parameters = {**parameters, ('hidden_width' if root == PARENT else 'hidden_widths'): (32 if root == PARENT else spec['hidden_widths'])}
    if any(state[k] != v for k, v in expected.items()) or state['parameters'] != parameters:
        raise ValueError('Model identity/recipe changed')
    if _json(folder / 'features.json') != parent.feature_names(spec):
        raise ValueError('Model feature ordering changed')
    return state


def predict(state, matrix, *, native=True):
    if state['schema_version'] == 'geo_helpfulness.joint_tcp_router_state.v1':
        from multimodal.geo_helpfulness_joint_tcp_fit import predict_joint
    else:
        from multimodal.geo_helpfulness_joint_tcp_capacity_fit import predict_joint
    return predict_joint(state, matrix, native=native)


def model_predictions(seed, spec, data, *, replay=True):
    state = model_state(seed, spec)
    root, folder = _state_folder(seed, spec)
    values = _load_arrays(folder / 'predictions.npz')
    for split in SPLITS:
        for key in ('row_uid', 'plot_idx', 'producer_fold'):
            if not np.array_equal(values[f'{split}_{key}'], data[f'{split}_{key}']):
                raise ValueError('Model prediction identity changed')
        if replay:
            native = predict(state, data[f"{split}_{spec['input_id']}"])
            for key in PREDICTION_KEYS:
                if values[f'{split}_{key}'].dtype != np.float32 or not np.array_equal(values[f'{split}_{key}'], native[key]):
                    raise ValueError('Exact native model replay failed')
    history = _json(folder / 'history.json')
    if len(history) != 300 or [r['epoch'] for r in history] != list(range(1, 301)):
        raise ValueError('Fixed budget changed')
    return {split: {key: values[f'{split}_{key}'] for key in PREDICTION_KEYS} for split in SPLITS}


def fit_seed(seed):
    from multimodal.geo_helpfulness_joint_tcp_capacity_fit import fit_joint
    cfg = validate_declaration()['config']
    data = load_inputs(seed)
    with _lock(OUTPUT, f'fit_{seed}'):
        if (OUTPUT / 'pauses').exists() and any((OUTPUT / 'pauses').iterdir()):
            raise RuntimeError('Preserved nonfinite/fit failure requires inspection')
        for spec in cfg['configs']:
            root, folder = _state_folder(seed, spec)
            if folder.exists():
                model_predictions(seed, spec, data)
                continue
            if spec['source'] == 'reuse_parent':
                raise ValueError('A sealed parent control is missing; never refit or modify the parent study')
            print(f"seed {seed}: fitting {spec['config_id']} (300 epochs)", flush=True)
            try:
                result = fit_joint(data[f"train_{spec['input_id']}"], data['y_train'], data['tcp_targets'],
                    {'validation': data[f"validation_{spec['input_id']}"]}, seed=seed,
                    config_id=spec['config_id'], auxiliary_weight=spec['auxiliary_weight'],
                    parameters={**cfg['parameters'], 'hidden_widths': spec['hidden_widths']})
            except Exception as error:
                paused = OUTPUT / 'pauses' / f"seed{seed}_{spec['config_id']}"
                paused.mkdir(parents=True, exist_ok=False)
                _write_json(paused / 'reason.json', {'reason': str(error), 'created_at': _now()})
                partial = getattr(error, 'partial_result', None)
                if partial:
                    for key in ('state', 'history', 'metadata'):
                        if key in partial:
                            _write_json(paused / f'{key}.json', partial[key])
                raise
            staged = _staging(OUTPUT, f"model_{seed}_{spec['config_id']}")
            for key in ('state', 'history', 'metadata'):
                _write_json(staged / f'{key}.json', result[key])
            _write_json(staged / 'features.json', parent.feature_names(spec))
            saved = {f'{split}_{key}': result['predictions'][split][key] for split in SPLITS for key in PREDICTION_KEYS}
            for split in SPLITS:
                for key in ('row_uid', 'plot_idx', 'producer_fold'):
                    saved[f'{split}_{key}'] = data[f'{split}_{key}']
            _save_arrays(staged / 'predictions.npz', saved)
            diagnostics = {}
            for split in SPLITS:
                portable = predict(result['state'], data[f"{split}_{spec['input_id']}"], native=False)
                diagnostics[split] = {key: float(np.abs(result['predictions'][split][key].astype(np.float64) - portable[key]).max()) for key in PREDICTION_KEYS}
            _write_json(staged / 'float64_replay.json', diagnostics)
            _seal_stage(staged, folder, OUTPUT, role='joint_capacity_model', metadata={
                'seed': seed, 'config_id': spec['config_id'], 'inputs_manifest_sha256': _hash(OUTPUT / f'inputs/seed_{seed}/manifest.json')})
            model_predictions(seed, spec, data)
    return {'status': 'complete', 'seed': seed, 'new_fits': 32, 'reused_fits': 8}


def prediction_frame(features, prediction, seed, spec, split):
    frame = pd.DataFrame({key: features[key] for key in ('row_uid', 'plot_idx', 'producer_fold')})
    frame['training_seed'], frame['split'] = seed, split
    for key in ('config_id', 'input_id', 'variant', 'with_embeddings', 'auxiliary_weight', 'architecture_id'):
        frame[key] = spec[key]
    for name in ('image', 'geo', 'raw'):
        frame[f'{name}_logits'] = list(features[f'{name}_logits'])
    for index, name in enumerate(TARGET_ORDER):
        frame[f'q_{name}'] = prediction['q'][:, index]
    frame['fusion_tcp_hat'], frame['geo_tcp_hat'] = prediction['tcp_hat'][:, 0], prediction['tcp_hat'][:, 1]
    for key in ('state_logits', 'tcp_logits'):
        frame[key] = list(prediction[key])
    return frame


def all_predictions(*, replay=True):
    frames, manifests = [], {}
    for seed in SEEDS:
        data = load_inputs(seed)
        for spec in configuration()['configs']:
            predictions = model_predictions(seed, spec, data, replay=replay)
            _, folder = _state_folder(seed, spec)
            manifests[str(folder / 'manifest.json')] = _hash(folder / 'manifest.json')
            for split in SPLITS:
                features = {key: data[f'{split}_{key}'] for key in ('row_uid', 'plot_idx', 'producer_fold', 'image_logits', 'geo_logits', 'raw_logits')}
                frames.append(prediction_frame(features, predictions[split], seed, spec, split))
    return pd.concat(frames, ignore_index=True), manifests


def save_report(staged, result, rows):
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import format_report
    _write_json(staged / 'report.json', result['report'])
    (staged / 'report.md').write_text(format_report(result['report']))
    _write_json(staged / 'selections.json', result.get('selections', {}))
    _write_json(staged / 'bootstrap_policy_order.json', result['bootstrap_policy_order'])
    _save_arrays(staged / 'bootstrap_draws.npz', {'draws': result['bootstrap_draws']})
    result['actions'].to_parquet(staged / 'actions.parquet', index=False)
    rows.to_parquet(staged / 'scored_rows.parquet', index=False)


def scored_rows():
    rows, manifests = all_predictions()
    verified_at = _now()
    path = PROTOCOL / 'protocol/development_assignments.parquet'
    declared = _json(OUTPUT / 'declaration.json')['source_file_sha256']
    if _hash(path) != declared[str(path.relative_to(ROOT))]:
        raise ValueError('Development assignments changed')
    labels = pd.read_parquet(path, columns=['row_uid', 'plot_idx', 'development_role', 'label_id_dense']).rename(columns={'development_role': 'split'})
    scored = rows.merge(labels, on=['row_uid', 'plot_idx', 'split'], how='left', validate='many_to_one', sort=False)
    if len(scored) != len(rows) or scored.label_id_dense.isna().any():
        raise ValueError('Scoring labels do not join uniquely')
    return scored, manifests, verified_at


def score():
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import make_development_report
    cfg = validate_declaration()['config']
    with _lock(OUTPUT, 'score'):
        folder = OUTPUT / 'scores'
        if folder.exists():
            _validate_stage(folder, OUTPUT, role='joint_capacity_scores')
            return {'status': 'reused_valid'}
        rows, manifests, verified_at = scored_rows()
        result = make_development_report(rows, cfg)
        # The reused control thresholds must remain byte-for-byte equivalent JSON values.
        controls = _json(PARENT / 'scores/selections.json')
        for name, selection in result['selections'].items():
            if name in controls and selection != controls[name]:
                raise ValueError(f'Reused control threshold changed: {name}')
        staged = _staging(OUTPUT, 'scores')
        save_report(staged, result, rows)
        _write_json(staged / 'prediction_verification.json', {'verified_at': verified_at, 'model_manifests': manifests})
        _seal_stage(staged, folder, OUTPUT, role='joint_capacity_scores', metadata={'model_manifests': manifests})
    return {'status': 'complete', 'policies': 401, 'contrasts': 1020}


def validate():
    from multimodal.geo_helpfulness_joint_tcp_capacity_report import make_development_report, format_report
    cfg = validate_declaration(deep=True)['config']
    rows, manifests, _ = scored_rows()
    folder = OUTPUT / 'scores'
    seal = _validate_stage(folder, OUTPUT, role='joint_capacity_scores')
    if seal['metadata']['model_manifests'] != manifests:
        raise ValueError('Scoring model binding changed')
    result = make_development_report(rows, cfg)
    if result['report'] != _json(folder / 'report.json') or result['selections'] != _json(folder / 'selections.json'):
        raise ValueError('Development metrics/selections do not reproduce')
    if format_report(result['report']) != (folder / 'report.md').read_text():
        raise ValueError('Development markdown does not reproduce')
    pd.testing.assert_frame_equal(rows, pd.read_parquet(folder / 'scored_rows.parquet'), check_exact=True)
    pd.testing.assert_frame_equal(result['actions'], pd.read_parquet(folder / 'actions.parquet'), check_exact=True)
    if result['bootstrap_policy_order'] != _json(folder / 'bootstrap_policy_order.json') or not np.array_equal(result['bootstrap_draws'], _load_arrays(folder / 'bootstrap_draws.npz')['draws']):
        raise ValueError('Bootstrap does not reproduce')
    return {'status': 'valid', 'new_fits': 128, 'reused_fits': 32, 'policies': 401, 'contrasts': 1020}
