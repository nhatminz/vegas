#!/usr/bin/env python3
"""Download original dataset files at an immutable HF revision; save Arrow too."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/mnt/hdd/nhatminh/data')
    args = parser.parse_args()
    for local, repo, filename in [('aime25', 'math-ai/aime25', 'test.jsonl'),
                                   ('codeelo', 'Qwen/CodeElo', 'test.json')]:
        info = json.load(urlopen(f'https://huggingface.co/api/datasets/{repo}'))
        root = Path(args.data_root) / local
        (root / 'raw').mkdir(parents=True, exist_ok=True)
        for name in (filename, 'README.md'):
            data = urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{info["sha"]}/{name}').read()
            (root / 'raw' / name).write_bytes(data)
        ds = load_dataset('json', data_files={'test': str(root / 'raw' / filename)})
        ds.save_to_disk(str(root / 'arrow'))
        manifest = {'repository': repo, 'revision': info['sha'], 'rows': len(ds['test']),
                    'features': ds['test'].features.to_dict(),
                    'sha256': hashlib.sha256((root / 'raw' / filename).read_bytes()).hexdigest()}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest))


if __name__ == '__main__':
    main()
