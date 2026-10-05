#!/usr/bin/env python3
"""Run the approved deeper joint router study and frozen legacy test transfer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multimodal import geo_helpfulness_joint_tcp_capacity as study
from multimodal import geo_helpfulness_joint_tcp_capacity_test as test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('declare', 'prepare', 'prepare-seed', 'fit', 'fit-seed', 'score', 'validate', 'test-declare', 'test-predict', 'test-score', 'test-validate', 'test-build', 'build'))
    parser.add_argument('--seed', type=int, choices=study.SEEDS)
    args = parser.parse_args()
    phases = []
    if args.command in ('prepare', 'build'):
        phases += [['declare']] + [['prepare-seed', '--seed', str(s)] for s in study.SEEDS]
    if args.command in ('fit', 'build'):
        phases += [['fit-seed', '--seed', str(s)] for s in study.SEEDS]
    if args.command == 'build':
        phases += [['score'], ['validate']]
    if args.command in ('build', 'test-build'):
        phases += [['test-declare']] + [['test-predict', '--seed', str(s)] for s in study.SEEDS]
        phases += [['test-score'], ['test-validate']]
    if phases:
        for phase in phases:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *phase], cwd=ROOT, check=True)
        result = {'status': 'complete', 'command': args.command}
    elif args.command in ('prepare-seed', 'fit-seed', 'test-predict'):
        if args.seed is None:
            parser.error('This stage requires --seed')
        fn = {'prepare-seed': study.prepare_seed, 'fit-seed': study.fit_seed, 'test-predict': test.predict_seed}[args.command]
        result = fn(args.seed)
    elif args.command.startswith('test-'):
        result = getattr(test, args.command[5:])()
    else:
        result = getattr(study, args.command)()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
