#!/usr/bin/env python3
"""Run GPU jobs serially; keep each process/log/result isolated."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['long', 'diagnostics', 'base', 'smoke'])
    args = parser.parse_args()
    jobs = []
    if args.phase == 'long':
        jobs = [(method, dataset, batch, 8192, 256, 3, [])
                for batch in [8, 16] for dataset in ['aime25', 'codeelo']
                for method in ['vllm', 'vegas']]
    elif args.phase == 'diagnostics':
        jobs = [(method, 'aime25', 8, 4096, 64, 1, ['--logprobs', '5', '--no-profile-overhead'])
                for method in ['vllm', 'vegas']]
        jobs += [('vegas', 'codeelo', 1, 4096, 64, 1,
                  ['--temperature', '0.6', '--top-p', '0.95', '--top-k', '20'])]
    elif args.phase == 'smoke':
        jobs = [(method, 'aime25', 1, 4096, 64, 1, []) for method in ['vllm', 'vegas']]
    else:
        jobs = [(method, dataset, batch, 4096, 64, 3, [])
                for batch in [8, 16] for dataset in ['aime25', 'codeelo']
                for method in ['vllm', 'vegas']]
    statuses = []
    for method, dataset, batch, length, tokens, rounds, extra in jobs:
        suffix = '_logprobs' if '--logprobs' in extra else '_random' if '--temperature' in extra else ''
        name = f'{method}_{dataset}_b{batch}_l{length}_n{tokens}{suffix}'
        if args.phase == 'smoke':
            name = f'smoke_{"baseline" if method == "vllm" else "vegas"}'
        script = 'run_baseline.sh' if method == 'vllm' else 'run_vegas.sh'
        cmd = [str(ROOT / 'scripts' / script), dataset, '--batch-size', str(batch),
               '--max-model-len', str(length), '--max-new-tokens', str(tokens),
               '--rounds', str(rounds), '--output', f'results/{name}', *extra]
        if args.phase == 'base':
            # These are the exact four batch-8 commands shown in the handoff.
            cmd = [str(ROOT / 'scripts' / script), dataset, '--batch-size', str(batch),
                   '--output', f'results/{name}']
        print('RUN', ' '.join(cmd), flush=True)
        start = time.monotonic()
        with (ROOT / 'results' / 'logs' / f'{name}.log').open('w') as log:
            process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {'name': name, 'command': cmd, 'returncode': process.returncode,
               'process_wall_s': time.monotonic() - start}
        statuses.append(row)
        (ROOT / 'results' / f'matrix_{args.phase}.json').write_text(json.dumps(statuses, indent=2) + '\n')
        print('DONE', name, 'exit', process.returncode, flush=True)
        if process.returncode:
            return process.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
