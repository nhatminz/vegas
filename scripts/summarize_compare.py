#!/usr/bin/env python3
"""Audit fair pairs and render measured JSON results; never fill missing data."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def main():
    rows, pairs = [], []
    fair_fields = ['model', 'batch_size', 'max_model_len', 'max_new_tokens', 'dtype',
                   'seed', 'temperature', 'top_p', 'top_k', 'rounds', 'warmup_rounds',
                   'thinking', 'offset', 'gpu_memory_utilization', 'max_num_batched_tokens']
    for length, tokens in [(4096, 64), (8192, 256)]:
        for batch in [8, 16]:
            for dataset in ['aime25', 'codeelo']:
                loaded = []
                for method in ['vllm', 'vegas']:
                    path = RESULTS / f'{method}_{dataset}_b{batch}_l{length}_n{tokens}.json'
                    result = json.loads(path.read_text())
                    assert result['status'] == 'PASS', path
                    assert all(r['output_tokens'] == tokens for r in result['requests']), path
                    assert result['metrics']['prefill_forward_stream_ms_per_request'] is not None, path
                    if method == 'vllm':
                        assert result['engine']['speculative_config'] is None
                        assert result['metrics']['acceptance_rate'] is None
                    else:
                        assert result['engine']['score_mode'] == 'logit'
                        ins = result['instrumentation']
                        assert ins['proposed_draft_tokens'] > 0
                        assert 0 <= ins['accepted_draft_tokens'] <= ins['proposed_draft_tokens']
                        assert all(not s['use_weight'] for s in ins['selection_samples'])
                    row = {'method': method, 'dataset': dataset, 'batch_size': batch,
                           'max_model_len': length, 'max_new_tokens': tokens,
                           'status': result['status'], **result['metrics'],
                           'json': path.name}
                    ins = result['instrumentation']
                    row['proposed_draft_tokens'] = ins['proposed_draft_tokens'] if method == 'vegas' else None
                    row['accepted_draft_tokens'] = ins['accepted_draft_tokens'] if method == 'vegas' else None
                    if method == 'vegas':
                        selected = [(b, n) for s in ins['selection_samples'] for b, n in zip(s['budget'], s['valid'])]
                        row['sparse_request_selection_samples'] = sum(b < n for b, n in selected)
                        row['selection_samples'] = len(selected)
                        row['min_effective_keep_ratio'] = min(b / n for b, n in selected if n)
                        row['max_effective_keep_ratio'] = max(b / n for b, n in selected if n)
                    rows.append(row)
                    loaded.append(result)
                baseline, vegas = loaded
                assert all(baseline['config'][k] == vegas['config'][k] for k in fair_fields)
                assert baseline['config'].get('progress', False) == vegas['config'].get('progress', False), (
                    'Progress setting differs; rerun both methods with the same --progress/--no-progress option')
                assert baseline['dataset']['prompt_sha256'] == vegas['dataset']['prompt_sha256']
                assert baseline['dataset']['sha256'] == vegas['dataset']['sha256']
                assert baseline['environment']['packages'] == vegas['environment']['packages']
                matches = sum(b['token_ids'] == v['token_ids'] for b, v in zip(baseline['outputs'], vegas['outputs']))
                pairs.append({'dataset': dataset, 'batch_size': batch, 'max_model_len': length,
                              'max_new_tokens': tokens, 'fair_configuration': True,
                              'greedy_identical_request_outputs': matches,
                              'request_outputs_compared': len(baseline['outputs']),
                              'vegas_e2e_speedup': vegas['metrics']['e2e_tokens_per_s'] / baseline['metrics']['e2e_tokens_per_s']})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (RESULTS / 'comparison_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / 'comparison_summary.json').write_text(json.dumps({'pairs': pairs, 'measurements': rows}, indent=2) + '\n')
    report = ['# Kết quả RTX 3090 thực đo', '',
              'Mỗi ô cấu hình có 1 batch warmup, 3 batch measurement và 1 batch profile riêng. BF16, greedy, seed 42, gamma 4, ratio 0.07, min KV 256, eager, FA2; cùng prompts/token IDs và sampling giữa hai phương pháp.', '',
              '**PASS ở đây xác nhận inference GPU và ghi metrics thành công. Không đồng nghĩa bitwise equivalence hoặc độ chính xác lời giải.** Prefill trong bảng là CUDA stream span target-forward theo request, không chia thời gian batch cho batch size. Prefill service, TTFT, queue và profile riêng nằm trong JSON/CSV.', '']
    for length, tokens in [(4096, 64), (8192, 256)]:
        report += [f'## max_model_len={length}, max_new_tokens={tokens}', '',
                   '| Dataset | Batch | Method | E2E tok/s | Decode tok/s | Accept % | Accepted/req verify | Prefill forward ms/req | TTFT ms/req | TPOT ms | Peak device MiB |',
                   '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['max_model_len'] != length:
                continue
            accept = 'N/A' if r['acceptance_rate'] is None else f"{100*r['acceptance_rate']:.2f}"
            accepted = 'N/A' if r['accepted_tokens_per_verification'] is None else f"{r['accepted_tokens_per_verification']:.3f}"
            report.append(f"| {r['dataset']} | {r['batch_size']} | [{r['method']}]({r['json']}) | {r['e2e_tokens_per_s']:.2f} | {r['decode_only_tokens_per_s']:.2f} | {accept} | {accepted} | {r['prefill_forward_stream_ms_per_request']:.2f} | {r['ttft_ms_per_request']:.2f} | {r['tpot_ms']:.3f} | {r['peak_vram_device_mib']:.1f} |")
        report += ['']
    report += ['## Kiểm tra so sánh', '',
               'Mọi cặp đã được assert cùng model, prompt hash, dataset hash, sampling, dtype, seed, context/token limit, concurrency, warmup/rounds và versions. Acceptance lấy từ nhánh accept thực tế của rejection kernel, không lấy output length; bonus/recovered token không tính accepted draft.', '',
               '| Dataset | Batch | Max len / new | Bitwise matched request outputs |',
               '|---|---:|---:|---:|']
    for p in pairs:
        report.append(f"| {p['dataset']} | {p['batch_size']} | {p['max_model_len']} / {p['max_new_tokens']} | {p['greedy_identical_request_outputs']} / {p['request_outputs_compared']} |")
    report += ['', 'BF16 không cho token equality ở mọi request. `greedy_precision_diagnostics.json` lưu first mismatch và target logprob margins; quan sát tie/top-1 đổi theo shape phù hợp với ảnh hưởng số học BF16, đây là giải thích dựa trên bằng chứng, không phải chứng minh cho mọi sai khác. Probe FP16 batch 8/64 trên cùng 8 bài AIME25 cho 8/8 token sequences giống nhau (`diagnostic_fp16_*.json`). Không tuyên bố bitwise losslessness cho mọi dtype/kernel/setting.', '',
               'Giới hạn: prompts thật của các batch này dài 49–850 token (AIME25) và 265–1249 token (CodeElo). `max_model_len=8192` là trần cấu hình; đây không phải benchmark prompt 8K thật. Mức sàn 256 khiến effective keep ratio cao hơn 0.07 ở context ngắn; số selection sparse và khoảng ratio thực nằm trong summary CSV.', '',
               'Peak VRAM là tổng device NVML lấy mẫu mỗi 10 ms trong measurement, có allocator/KV reservation và display. PyTorch allocated/reserved peak lưu riêng. Decode-only bắt đầu sau first-token của request cuối cùng; TPOT tính từng request, gồm thời gian chờ khi có mixed prefill/decode. Profile per-layer chạy ở batch riêng, không nằm trong throughput measurement.', '',
               'Không OOM ở các cấu hình PASS liệt kê. Hai lỗi debug ban đầu đã được sửa: CUB workspace query và hidden-state gather cố gọi nvcc không tồn tại; giữ log trong `logs/*attempt01.log` và failure JSON ở `failures/`. B200 chưa được kiểm thử.', '']
    (RESULTS / 'RESULTS.md').write_text('\n'.join(report))
    print(f'Audited {len(pairs)} fair pairs, wrote {len(rows)} measured rows.')


if __name__ == '__main__':
    main()
