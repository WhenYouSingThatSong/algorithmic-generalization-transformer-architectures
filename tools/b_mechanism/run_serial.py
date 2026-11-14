#!/usr/bin/env python3
"""Run an explicit JSON job list serially; never discovers or starts jobs itself.

Job file: {"num_threads": 4, "jobs": [{"argv": ["tools/...py", "--out", "..."],
"gpu": "1", "log": "tmp/...log"}]}. argv excludes the Python executable.
Run from repository root. Stops on first failed job, does not manage training.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.jobs).read_text(encoding='utf-8'))
    threads = int(spec.get('num_threads', 4))
    if threads < 1:
        parser.error('num_threads must be positive')
    for i, job in enumerate(spec['jobs'], 1):
        argv = list(job['argv'])
        if not argv or not all(isinstance(x, str) for x in argv):
            parser.error('each job.argv must be a nonempty string list')
        if not any(x == '--num_threads' or x.startswith('--num_threads=') for x in argv):
            argv.extend(['--num_threads', str(threads)])
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(job['gpu'])
        env['PYTHONUNBUFFERED'] = '1'
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS'):
            env[key] = str(threads)
        log = Path(job['log'])
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', *argv]
        print(json.dumps(dict(job=i, command=command, gpu=job['gpu'], log=str(log))), flush=True)
        # New logs only: preserve failed attempts and earlier experiment records.
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'job {i} failed ({result.returncode}); see {log}; remaining jobs not started')
    print('Serial queue completed.', flush=True)


if __name__ == '__main__':
    main()
