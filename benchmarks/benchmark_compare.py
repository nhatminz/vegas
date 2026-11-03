#!/usr/bin/env python3
"""Fair offline Qwen3 baseline / Vegas benchmark on a single GPU.

Use scripts/run_{baseline,vegas}.sh so offline/cache settings are consistent.
Each process owns one eager vLLM engine; warmup is excluded from every metric.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['vllm', 'vegas'], required=True)
    p.add_argument('--dataset', choices=['aime25', 'codeelo'], required=True)
    p.add_argument('--model', default='/mnt/hdd/nhatminh/SpecDecode/models/Qwen3-4B')
    p.add_argument('--data-root', default='/mnt/hdd/nhatminh/data')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gamma', type=int, default=4)
    p.add_argument('--sparsity-ratio', type=float, default=0.07)
    p.add_argument('--min-kv-tokens', type=int, default=256)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--max-num-batched-tokens', type=int, default=4096)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    p.add_argument('--dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--logprobs', type=int, default=0, help='Optional top-N target logprobs for correctness diagnostics')
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--warmup-rounds', type=int, default=1)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--profile-overhead', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True,
                   help='Show tqdm stages and real emitted output-token progress (default: enabled)')
    p.add_argument('--print-json', action='store_true',
                   help='Print raw JSON instead of the readable summary; result files always contain JSON/CSV')
    p.add_argument('--output', help='Result prefix, no extension')
    a = p.parse_args()
    if min(a.batch_size, a.gamma, a.rounds, a.max_new_tokens) < 1 or a.warmup_rounds < 1:
        p.error('batch/gamma/rounds/tokens must be positive and at least one warmup is required')
    return a


def provenance():
    import torch
    import vllm
    import triton
    from vllm.vllm_flash_attn import __version__ as fa_version
    from vllm.vllm_flash_attn import flash_attn_interface as fa
    packages = {}
    for name in ['torch', 'vllm', 'triton', 'transformers', 'datasets', 'flash-attn', 'flashinfer-python']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    hashes = {}
    patterns = ['benchmarks/compare*.py', 'benchmarks/benchmark_compare.py',
                'vllm/v1/spec_decode/sparse_attn/**/*.py', 'scripts/run*.sh',
                'vllm/v1/sample/rejection_sampler.py']
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    gpu = torch.cuda.get_device_properties(0)
    return {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'upstream_vllm_commit': '89a77b10846fd96273cce78d86d2556ea582d26e',
            'modified_flash_attention_commit': '0c05c4b59b4a795f2c2add264fef094e758478e3',
            'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
            'source_sha256': hashes, 'python': sys.version, 'packages': packages,
            'vllm_import_path': vllm.__file__, 'torch_cuda': torch.version.cuda,
            'gpu': gpu.name, 'compute_capability': [gpu.major, gpu.minor],
            'gpu_memory_bytes': gpu.total_memory,
            'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
            'nvcc': None, 'triton': triton.__version__, 'torch_version': torch.__version__,
            'flash_attention': {'fa2_available': fa.FA2_AVAILABLE,
                                'fa3_available': fa.FA3_AVAILABLE,
                                'python_package_version': fa_version,
                                'modified_fa3_built_or_used': False,
                                'actual_verify_version': 2,
                                'extension_source': 'exact upstream vLLM precompiled wheel; unmodified FA2',
                                'vegas_backend': os.environ.get('VEGAS_BACKEND', 'auto')}}


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def run(a, result):
    with ExitStack() as resources:
        _run(a, result, resources)


def _shutdown_distributed(result):
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel,
    )
    try:
        destroy_model_parallel()
    finally:
        destroy_distributed_environment()
    result.setdefault('cleanup', {})['distributed_shutdown'] = True


def _shutdown_engine(llm, result):
    llm.llm_engine.engine_core.shutdown()
    result.setdefault('cleanup', {})['engine_shutdown'] = True


def _run(a, result, resources):
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from compare_data import rows_and_prompts
    from compare_measurement import Measurement, MemoryMonitor
    from compare_progress import BenchmarkProgress
    from vllm.v1.spec_decode.sparse_attn.attn_overrider.vegas import VegasAttnOverrider

    torch.manual_seed(a.seed)
    result['environment'] = provenance()
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    rows, all_prompts, dataset = rows_and_prompts(a.dataset, a.data_root, tokenizer, a.thinking)
    # Reserve the same gamma margin for BOTH methods; never truncate silently.
    eligible = [i for i, ids in enumerate(all_prompts)
                if len(ids) + a.max_new_tokens + a.gamma <= a.max_model_len]
    indices = eligible[a.offset:a.offset + a.batch_size]
    if len(indices) != a.batch_size:
        raise ValueError(f'Only {len(indices)} eligible prompts available for requested batch')
    prompts = [{'prompt_token_ids': all_prompts[i]} for i in indices]
    result['dataset'] = {**dataset, 'indices': indices,
                         'ids': [rows[i].get('id', rows[i].get('problem_id')) for i in indices],
                         'prompt_tokens': [len(all_prompts[i]) for i in indices],
                         'prompt_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                         'overlength_rows_skipped': len(rows) - len(eligible),
                         'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    result['model_files'] = {
        path.name: {'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.suffix == '.json' else None}
        for path in Path(a.model).iterdir() if path.is_file()}
    speculative = None if a.method == 'vllm' else {
        'method': 'sparse_attn', 'num_speculative_tokens': a.gamma,
        'sparse_attn_algorithm': 'vegas', 'sparse_attn_ratio': a.sparsity_ratio,
        'sparse_attn_min_tokens': a.min_kv_tokens}
    assert VegasAttnOverrider.SCORE_MODE == 'logit'
    result['engine'] = {'speculative_config': json.loads(json.dumps(speculative)), 'score_mode': 'logit' if speculative else None,
                        'enforce_eager': True, 'async_scheduling': False,
                        'enable_prefix_caching': False, 'flash_attn_version': 2}
    m = Measurement()
    m.install()
    # Register before constructing LLM: initialization errors may already have
    # created process groups. ExitStack still cleans them up on exceptions.
    resources.callback(_shutdown_distributed, result)
    print(f'Initializing {a.method} / {a.dataset}, batch={a.batch_size} ...', flush=True)
    started = time.perf_counter()
    llm = LLM(model=a.model, tokenizer=a.model, dtype=a.dtype, seed=a.seed,
              tensor_parallel_size=1, max_model_len=a.max_model_len,
              max_num_seqs=a.batch_size, max_num_batched_tokens=a.max_num_batched_tokens,
              gpu_memory_utilization=a.gpu_memory_utilization,
              speculative_config=speculative, enforce_eager=True,
              compilation_config={'mode': 0, 'cudagraph_mode': 'NONE'},
              attention_config={'backend': 'FLASH_ATTN', 'flash_attn_version': 2},
              enable_prefix_caching=False, enable_chunked_prefill=True,
              async_scheduling=False, disable_log_stats=False)
    resources.callback(_shutdown_engine, llm, result)
    result['initialization_s'] = time.perf_counter() - started
    progress = resources.enter_context(BenchmarkProgress(
        enabled=a.progress, stages=a.warmup_rounds + a.rounds + int(a.profile_overhead),
        tokens_per_batch=a.batch_size * a.max_new_tokens,
        description=f'Benchmark {a.method}/{a.dataset}'))
    params = SamplingParams(max_tokens=a.max_new_tokens, temperature=a.temperature,
                            top_p=a.top_p, top_k=a.top_k, ignore_eos=True,
                            logprobs=a.logprobs or None,
                            presence_penalty=0.0, frequency_penalty=0.0,
                            repetition_penalty=1.0)
    for rep in range(a.warmup_rounds):
        with progress.phase(m, f'Warmup {rep + 1}/{a.warmup_rounds}'):
            llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    m.reset()
    m.active = True
    request_rows, output_rows, batches = [], [], []
    with MemoryMonitor() as memory:
        for rep in range(a.rounds):
            with progress.phase(m, f'Measurement {rep + 1}/{a.rounds}'):
                emit_start = len(m.emissions)
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                wall = time.perf_counter() - start
            assert len(outputs) == a.batch_size
            first, last = [], []
            for i, out in enumerate(outputs):
                tokens = list(out.outputs[0].token_ids)
                assert len(tokens) == a.max_new_tokens, (len(tokens), a.max_new_tokens)
                stats = out.metrics
                if stats is None or not stats.first_token_ts or not stats.scheduled_ts:
                    raise RuntimeError('Native request timestamps missing; latency cannot be fabricated')
                first.append(stats.first_token_ts)
                last.append(stats.last_token_ts)
                dt = stats.last_token_ts - stats.first_token_ts
                request_rows.append({'round': rep, 'request_id': out.request_id,
                    'dataset_index': indices[i], 'prompt_tokens': len(out.prompt_token_ids),
                    'output_tokens': len(tokens), 'ttft_ms': stats.first_token_latency * 1000,
                    'prefill_service_ms': (stats.first_token_ts - stats.scheduled_ts) * 1000,
                    'queue_ms': (stats.scheduled_ts - stats.queued_ts) * 1000,
                    'decode_service_ms': dt * 1000,
                    'tpot_ms': dt * 1000 / (len(tokens) - 1) if len(tokens) > 1 else None,
                    'native_stats': asdict(stats)})
                output_rows.append({'round': rep, 'dataset_index': indices[i],
                                    'token_ids': tokens, 'text': out.outputs[0].text})
                if a.logprobs:
                    output_rows[-1]['logprobs'] = [
                        {str(k): asdict(v) for k, v in values.items()}
                        for values in out.outputs[0].logprobs]
            # All-prefills-complete barrier: count only emission chunks strictly
            # after it, so mixed prefill/decode time is excluded from decode-only.
            barrier = max(first)
            decode_s = max(last) - barrier
            decode_tokens = sum(n for _, ts, n in m.emissions[emit_start:]
                                if ts is not None and ts > barrier)
            batches.append({'round': rep, 'wall_s': wall, 'output_tokens': sum(len(o.outputs[0].token_ids) for o in outputs),
                            'steady_decode_s': decode_s, 'steady_decode_tokens': decode_tokens})
    m.active = False
    instrumentation = m.summarize()
    if a.method == 'vegas':
        if not instrumentation['proposed_draft_tokens'] or not instrumentation['selection_samples']:
            raise RuntimeError('Vegas did not collect scores and verify any real draft tokens')
        assert not any(x['use_weight'] for x in instrumentation['selection_samples'])
    for row in request_rows:
        row['prefill_forward_stream_ms'] = instrumentation['prefill_forward_ms_by_request'].get(row['request_id'])
    wall = sum(x['wall_s'] for x in batches)
    ds = sum(x['steady_decode_s'] for x in batches)
    result.update(status='PASS', batches=batches, requests=request_rows, outputs=output_rows,
                  instrumentation=instrumentation)
    result['metrics'] = {
        'e2e_tokens_per_s': sum(x['output_tokens'] for x in batches) / wall,
        'decode_only_tokens_per_s': sum(x['steady_decode_tokens'] for x in batches) / ds if ds > 0 else None,
        'acceptance_rate': instrumentation['acceptance_rate'] if speculative else None,
        'accepted_tokens_per_verification': instrumentation['accepted_tokens_per_verification'] if speculative else None,
        'proposed_draft_tokens': instrumentation['proposed_draft_tokens'] if speculative else None,
        'accepted_draft_tokens': instrumentation['accepted_draft_tokens'] if speculative else None,
        'request_verification_iterations': instrumentation['request_verification_iterations'] if speculative else None,
        'prefill_service_ms_per_request': mean([r['prefill_service_ms'] for r in request_rows]),
        'prefill_forward_stream_ms_per_request': mean([r['prefill_forward_stream_ms'] for r in request_rows]),
        'ttft_ms_per_request': mean([r['ttft_ms'] for r in request_rows]),
        'tpot_ms': mean([r['tpot_ms'] for r in request_rows]),
        'peak_vram_device_mib': memory.peak / 2**20,
        'peak_torch_allocated_mib': memory.allocated / 2**20,
        'peak_torch_reserved_mib': memory.reserved / 2**20}
    result['metric_notes'] = {
        'e2e': 'Output tokens / measured generate wall time. Model load, tokenization and warmup excluded; scheduling, prefill, decode included.',
        'decode_only': 'Emission tokens strictly after all requests have first tokens / time from latest first-token timestamp to last emission; sum across rounds.',
        'prefill_service': 'Per-request native scheduled-to-first-emission service span; includes sampling and initial Vegas draft; queue is excluded.',
        'prefill_forward_stream': 'CUDA-event target-forward spans for each request while its prompt is processed; shared batch service time is NOT divided by batch.',
        'tpot': 'Per-request (last emission - first emission) / (output tokens - 1); speculative multi-token bursts are amortized, not individual token arrival intervals.',
        'overhead': 'Nested CUDA stream spans include launch gaps. target_forward includes collection/reduce/topk; draft includes slot mapping. Rejection excludes recomputed draft probabilities; sampling_and_draft_probs includes them. Do not sum overlapping spans.',
        'acceptance': 'Direct counters in greedy/random rejection kernel accept branches, before max_tokens/EOS truncation. Denominator is drafts actually passed to verification; unused final drafts excluded. Accepted tokens per verification is per REQUEST iteration, excludes recovered/bonus tokens; batch iteration count reported separately.',
        'memory': 'Peak total device NVML sampled every 10ms during measurement, includes display/allocator/KV reservation. PyTorch peaks separately reported.',
        'progress': 'tqdm counts real emitted output tokens for warmup/measurement/profile separately. Console update cost during generate is included; use --no-progress on BOTH methods for runs without display overhead.',
        'limits': 'Eager, synchronous, single GPU; no penalties/bad words/custom logits processors. Dataset benchmark measures speed, not task accuracy.'}
    if a.profile_overhead:
        # Detailed per-layer events are intentionally outside throughput,
        # latency, acceptance totals and memory measurement.
        m.reset()
        m.enabled_labels = None
        m.active = True
        with progress.phase(m, 'Profile 1/1 (excluded from throughput)'):
            start = time.perf_counter()
            llm.generate(prompts, params, use_tqdm=False)
            torch.cuda.synchronize()
            profile_wall = time.perf_counter() - start
        m.active = False
        profile = m.summarize()
        result['overhead_profile'] = {
            'separate_round': True, 'wall_s': profile_wall,
            'stream_spans': profile['stream_spans'],
            'note': 'Separate profiled batch, excluded from primary metrics; event overhead can inflate spans.'}
        for label in ['draft', 'target_decode_forward', 'rejection_sampler',
                      'logit_collection', 'score_reduction', 'kv_topk', 'draft_slot_mapping']:
            span = profile['stream_spans'].get(label)
            result['metrics'][f'profile_{label}_ms_per_call'] = (
                span['total_ms'] / span['calls'] if span and span['calls'] else None)


def print_summary(a, result, prefix):
    print(f'\nBenchmark {result["status"]}: {a.method} / {a.dataset}, batch={a.batch_size}')
    if result['status'] != 'PASS':
        print(f'Error: {result.get("error", "see traceback in JSON")}')
    else:
        metrics = result['metrics']
        def value(name, unit):
            number = metrics.get(name)
            return f'{number:.3f} {unit}' if number is not None else 'N/A'
        for label, name, unit in [
            ('E2E throughput', 'e2e_tokens_per_s', 'tok/s'),
            ('Decode-only throughput', 'decode_only_tokens_per_s', 'tok/s'),
            ('Prefill forward', 'prefill_forward_stream_ms_per_request', 'ms/request'),
            ('Prefill service', 'prefill_service_ms_per_request', 'ms/request'),
            ('TTFT', 'ttft_ms_per_request', 'ms/request'),
            ('TPOT', 'tpot_ms', 'ms/token'),
            ('Peak device VRAM', 'peak_vram_device_mib', 'MiB'),
        ]:
            print(f'  {label:<25} {value(name, unit)}')
        if a.method == 'vllm':
            print('  Acceptance / draft / KV selection: N/A (autoregressive baseline; no speculation)')
        else:
            print(f'  Acceptance rate           {100 * metrics["acceptance_rate"]:.2f}%')
            print(f'  Accepted / proposed       {metrics["accepted_draft_tokens"]} / {metrics["proposed_draft_tokens"]}')
            print(f'  Accepted / request verify {metrics["accepted_tokens_per_verification"]:.3f}')
        if a.profile_overhead:
            print('  Separate profile (excluded from throughput):')
            for label, name in [
                ('Target decode forward', 'target_decode_forward'),
                ('Draft', 'draft'), ('Rejection sampler', 'rejection_sampler'),
                ('Raw QK collection', 'logit_collection'),
                ('Score reduction', 'score_reduction'), ('KV top-k', 'kv_topk'),
                ('Draft slot mapping', 'draft_slot_mapping'),
            ]:
                key = f'profile_{name}_ms_per_call'
                if metrics.get(key) is not None:
                    print(f'    {label:<23} {value(key, "ms/call")}')
        else:
            print('  Detailed overhead: N/A (--no-profile-overhead)')
    print(f'JSON: {prefix.with_suffix(".json").resolve()}')
    print(f'CSV:  {prefix.with_suffix(".csv").resolve()}')


def main():
    a = arguments()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix = Path(a.output or ROOT / 'results' / f'{stamp}_{a.method}_{a.dataset}_b{a.batch_size}_l{a.max_model_len}_n{a.max_new_tokens}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'FAIL', 'timestamp_utc': stamp, 'config': vars(a), 'command': sys.argv}
    try:
        run(a, result)
    except Exception as error:
        result['status'] = 'FAIL'
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
        flat = {'status': result['status'], 'method': a.method, 'dataset': a.dataset,
                'batch_size': a.batch_size, 'max_model_len': a.max_model_len,
                'max_new_tokens': a.max_new_tokens, 'gamma': a.gamma,
                'sparsity_ratio': a.sparsity_ratio, 'seed': a.seed,
                **result.get('metrics', {}), 'json': str(prefix.with_suffix('.json'))}
        with prefix.with_suffix('.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat)
            writer.writeheader()
            writer.writerow(flat)
        if result.get('requests'):
            fields = [k for k in result['requests'][0] if k != 'native_stats']
            with prefix.with_name(prefix.name + '_requests').with_suffix('.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(result['requests'])
        if a.print_json:
            print(json.dumps(flat, indent=2))
        else:
            print_summary(a, result, prefix)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
