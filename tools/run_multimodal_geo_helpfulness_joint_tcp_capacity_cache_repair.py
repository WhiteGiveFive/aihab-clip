#!/usr/bin/env python3
"""Additive fix for the NPZ `file` member collision in the sealed test writer.

Original fitting and application sources remain unchanged. This helper seals an
explicit serialization amendment before creating any test prediction bundle.
The only change is how the image-path array is written to the NPZ archive.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multimodal import geo_helpfulness_joint_tcp_capacity_test as application
from multimodal import geo_helpfulness_joint_tcp_capacity as study
from multimodal.geo_helpfulness_confidnet import (
    _hash, _json, _load_arrays, _lock, _now, _save_arrays, _seal_stage, _staging, _write_json,
)
from multimodal.geo_helpfulness_joint_tcp_capacity_report import policy_actions
from multimodal.geo_helpfulness_joint_tcp_capacity_test_inputs import prepare_seed

TOOL = Path(__file__).resolve()
AMENDMENT_ID = 'npz_file_member_serialization_v1'


def save_features(path, arrays):
    """Keep the literal `file` member without colliding with NumPy's argument."""
    remaining = dict(arrays)
    paths = remaining.pop('file')
    paths = np.asarray(paths)
    if paths.ndim != 1 or paths.dtype.kind not in 'US':
        raise ValueError('Image paths must be a one-dimensional string array')
    _save_arrays(path, remaining)
    buffer = io.BytesIO()
    np.save(buffer, paths, allow_pickle=False)
    with zipfile.ZipFile(path, mode='a', compression=zipfile.ZIP_DEFLATED) as archive:
        if 'file.npy' in archive.namelist():
            raise ValueError('The NPZ image-path member already exists')
        archive.writestr('file.npy', buffer.getvalue())
    saved = _load_arrays(path)
    if set(saved) != set(arrays) or any(saved[k].dtype != arrays[k].dtype or not np.array_equal(saved[k], arrays[k]) for k in arrays):
        raise ValueError('The compatibility writer changed a feature value or dtype')


def validate_amendment():
    declaration = application.validate_declaration()
    record = declaration.get('serialization_amendment')
    if not isinstance(record, dict) or record.get('id') != AMENDMENT_ID:
        raise ValueError('Test declaration has no matching serialization amendment')
    original = application.OUTPUT / record['failed_declaration_path']
    if (record['helper_file_sha256'] != _hash(TOOL)
            or record['original_declaration_sha256'] != _hash(original)
            or declaration['source_file_sha256'].get(str(TOOL.relative_to(ROOT))) != _hash(TOOL)):
        raise ValueError('Serialization amendment source or preserved attempt changed')
    old = _json(original)
    for key, value in old.items():
        if key != 'source_file_sha256' and declaration.get(key) != value:
            raise ValueError('Serialization amendment changed a scientific declaration field')
    if {k: v for k, v in declaration['source_file_sha256'].items() if k != str(TOOL.relative_to(ROOT))} != old['source_file_sha256']:
        raise ValueError('Serialization amendment changed a previous source binding')
    if study.snapshot(original.parent) != record['failed_attempt_snapshot']:
        raise ValueError('Preserved failed-attempt artifacts changed')
    return declaration


def declare():
    with _lock(application.OUTPUT, 'serialization_amendment'):
        old = _json(application.OUTPUT / 'declaration.json')
        if old.get('serialization_amendment'):
            validate_amendment()
            return {'status': 'reused_valid'}
        application.validate_declaration()
        forbidden = ('predictions', 'scores', 'label_acquisition')
        if any((application.OUTPUT / name).exists() for name in forbidden):
            raise ValueError('Cannot amend after predictions or scoring have been sealed')
        failed = application.OUTPUT / 'failed_attempts/npz_file_keyword'
        failed.mkdir(parents=True, exist_ok=False)
        original_path = application.OUTPUT / 'declaration.json'
        original_hash = _hash(original_path)
        original_path.rename(failed / 'declaration.json')
        if (application.OUTPUT / 'staging').exists():
            (application.OUTPUT / 'staging').rename(failed / 'staging')
        amended = dict(old)
        amended['source_file_sha256'] = {**old['source_file_sha256'], str(TOOL.relative_to(ROOT)): _hash(TOOL)}
        amended['serialization_amendment'] = {
            'id': AMENDMENT_ID, 'created_at': _now(),
            'reason': 'np.savez_compressed output argument collides with feature array named file',
            'failed_declaration_path': str((failed / 'declaration.json').relative_to(application.OUTPUT)),
            'original_declaration_sha256': original_hash, 'helper_file_sha256': _hash(TOOL),
            'failed_attempt_snapshot': study.snapshot(failed),
            'scientific_recipe_changed': False, 'models_changed': False,
            'thresholds_changed': False, 'test_labels_accessed_before_amendment': False,
            'native_inference_changed': False,
        }
        _write_json(application.OUTPUT / 'declaration.json', amended)
        validate_amendment()
    return {'status': 'declared', 'amendment': AMENDMENT_ID}


def predict_seed(seed):
    declaration = validate_amendment()
    with _lock(application.OUTPUT, f'predict_{seed}'):
        final = application.OUTPUT / f'predictions/seed_{seed}'
        if final.exists():
            application.predictions(seed, replay=True)
            return {'status': 'reused_valid', 'seed': seed}
        prepared = prepare_seed(seed, context=declaration['legacy_context'])
        features = prepared['test']
        rows, diagnostics = application._infer(seed, features, declaration['config'])
        actions = policy_actions(rows, declaration['config'], declaration['validation_selections'])
        if any('label' in key or key == 'target_state' for key in (*rows.columns, *actions.columns)):
            raise ValueError('Prediction bundle contains labels')
        staged = _staging(application.OUTPUT, f'predictions_{seed}_npz_compat')
        rows.to_parquet(staged / 'rows.parquet', index=False)
        actions.to_parquet(staged / 'actions.parquet', index=False)
        save_features(staged / 'features.npz', features)
        _write_json(staged / 'source_replay.json', prepared['metadata'])
        _write_json(staged / 'float64_replay.json', diagnostics)
        _write_json(staged / 'serialization_receipt.json', declaration['serialization_amendment'])
        _seal_stage(staged, final, application.OUTPUT, role='joint_tcp_cleaned_test_predictions', metadata={'seed': seed})
        application.predictions(seed, replay=True)
    return {'status': 'complete', 'seed': seed, 'routers': 40, 'amendment': AMENDMENT_ID}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('declare', 'predict', 'validate', 'build'))
    parser.add_argument('--seed', type=int, choices=study.SEEDS)
    args = parser.parse_args()
    if args.command == 'build':
        runner = ROOT / 'tools/run_multimodal_geo_helpfulness_joint_tcp_capacity.py'
        for phase in ('prepare', 'fit', 'score', 'validate', 'test-declare'):
            subprocess.run([sys.executable, str(runner), phase], cwd=ROOT, check=True)
        declare()
        for seed in study.SEEDS:
            predict_seed(seed)
        for phase in ('test-score', 'test-validate'):
            subprocess.run([sys.executable, str(runner), phase], cwd=ROOT, check=True)
        validate_amendment()
        result = {'status': 'complete', 'command': 'build', 'amendment': AMENDMENT_ID}
    elif args.command == 'declare':
        result = declare()
    elif args.command == 'validate':
        validate_amendment()
        result = {'status': 'valid'}
    else:
        if args.seed is None:
            parser.error('predict requires --seed')
        result = predict_seed(args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
