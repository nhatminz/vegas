"""Offline dataset parsing and Qwen3 chat prompts shared by all methods."""
import hashlib
import json
from pathlib import Path


def rows_and_prompts(name, data_root, tokenizer, thinking=True):
    path = Path(data_root) / name / 'raw' / ('test.jsonl' if name == 'aime25' else 'test.json')
    raw = path.read_bytes()
    rows = [json.loads(x) for x in raw.decode().splitlines() if x.strip()] if name == 'aime25' else json.loads(raw)
    prompts = []
    for row in rows:
        if name == 'aime25':
            assert isinstance(row['problem'], str) and isinstance(row['answer'], (int, str))
            content = row['problem'] + '\nPlease reason carefully and put the final answer in \\boxed{}.'
        else:
            for field in ('title', 'description', 'input', 'output', 'interaction', 'note'):
                assert isinstance(row[field], str), field
            assert all(isinstance(x, list) and len(x) == 2 for x in row['examples'])
            parts = [row['title'], row['description'], 'Input\n' + row['input'], 'Output\n' + row['output']]
            if row['interaction']:
                parts += ['Interaction\n' + row['interaction']]
            for i, (inp, out) in enumerate(row['examples'], 1):
                parts += [f'Example {i}\nInput:\n{inp}\nOutput:\n{out}']
            if row['note']:
                parts += ['Note\n' + row['note']]
            parts += ['Write a complete solution in C++17. Explain your reasoning, then provide the code.']
            content = '\n\n'.join(parts)
        ids = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking)
        prompts.append(ids)
    return rows, prompts, {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'rows': len(rows), 'schema': {k: type(v).__name__ for k, v in rows[0].items()}}
