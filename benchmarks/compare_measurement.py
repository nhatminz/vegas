"""Opt-in eager/in-process instrumentation, shared by baseline and Vegas.

CUDA events measure stream spans, including host launch gaps. Nested spans
overlap; do not add draft/target/selection/collection/rejection totals together.
No synchronization is inserted between layers or speculative iterations.
"""
from collections import defaultdict
from functools import wraps
import threading
import time

import torch


class Measurement:
    def __init__(self):
        self.active = False
        self.progress_callback = None
        self.enabled_labels = {'target_forward'}
        self.reset()

    def reset(self):
        self.events = defaultdict(list)
        self.acceptance = []
        self.selections = []
        self.emissions = []
        self.last_counts = {}
        self.prompt_remaining = {}
        self.prefill_events = defaultdict(list)
        self.external_ids = {}
        self.raw_score_probe = None

    def timed(self, label, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.active or (self.enabled_labels is not None and label not in self.enabled_labels):
                return fn(*args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            self.events[label].append((start, end))
            return result
        return wrapper

    def observe_acceptance(self, proposed, accepted):
        if self.active:
            self.acceptance.append((list(proposed), accepted))

    def install(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.sample.rejection_sampler import RejectionSampler
        from vllm.v1.spec_decode.sparse_attn.proposer import SparseAttnProposer
        from vllm.v1.spec_decode.sparse_attn.attn_overrider.vegas import VegasAttnOverrider
        from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils import sm86
        from vllm.v1.engine.output_processor import OutputProcessor

        original_init = RejectionSampler.__init__
        def init(sampler, *args, **kwargs):
            original_init(sampler, *args, **kwargs)
            sampler.acceptance_observer = self.observe_acceptance
        RejectionSampler.__init__ = init
        RejectionSampler.forward = self.timed('rejection_sampler', RejectionSampler.forward)
        SparseAttnProposer.propose = self.timed('draft', SparseAttnProposer.propose)
        GPUModelRunner._sample = self.timed('sampling_and_draft_probs', GPUModelRunner._sample)
        GPUModelRunner._model_forward = self.timed('target_forward', GPUModelRunner._model_forward)
        VegasAttnOverrider._init_metadata = self.timed('draft_slot_mapping', VegasAttnOverrider._init_metadata)
        sm86.collect_raw_scores = self.timed('logit_collection', sm86.collect_raw_scores)
        sm86.reduce_logit_scores = self.timed('score_reduction', sm86.reduce_logit_scores)
        sm86.request_topk = self.timed('kv_topk', sm86.request_topk)

        verify = VegasAttnOverrider._verify_attention
        def observe_selection(overrider, *args, **kwargs):
            result = verify(overrider, *args, **kwargs)
            if self.active and overrider.curr_layer == 0:
                n = overrider.batch_size
                if self.raw_score_probe is None and overrider.portable:
                    self.raw_score_probe = overrider._attn_score_buffer[0, 0, 1, :16].clone()
                self.selections.append((overrider._topk_budget[:n].clone(),
                                        overrider._topk_valid_lens[:n].clone(),
                                        overrider._use_weight))
            return result
        VegasAttnOverrider._verify_attention = observe_selection

        execute = GPUModelRunner.execute_model
        def execute_model(runner, scheduler_output, *args, **kwargs):
            if not self.active:
                return execute(runner, scheduler_output, *args, **kwargs)
            for req in scheduler_output.scheduled_new_reqs:
                self.prompt_remaining[req.req_id] = len(req.prompt_token_ids)
            prefill = []
            for rid, count in scheduler_output.num_scheduled_tokens.items():
                if self.prompt_remaining.get(rid, 0) > 0:
                    prefill.append(rid)
                    self.prompt_remaining[rid] = max(0, self.prompt_remaining[rid] - count)
            before = len(self.events['target_forward'])
            result = execute(runner, scheduler_output, *args, **kwargs)
            event_pairs = self.events['target_forward'][before:]
            kind = 'target_prefill_forward' if prefill else 'target_decode_forward'
            self.events[kind].extend(event_pairs)
            for rid in prefill:
                self.prefill_events[rid].extend(event_pairs)
            return result
        GPUModelRunner.execute_model = execute_model

        process = OutputProcessor.process_outputs
        def process_outputs(processor, engine_core_outputs, engine_core_timestamp=None, **kwargs):
            if self.progress_callback is not None:
                self.progress_callback(sum(len(out.new_token_ids) for out in engine_core_outputs))
            if self.active:
                for out in engine_core_outputs:
                    state = processor.request_states.get(out.request_id)
                    if state is not None:
                        self.external_ids[out.request_id] = state.external_req_id
                    self.emissions.append((out.request_id, engine_core_timestamp, len(out.new_token_ids)))
            return process(processor, engine_core_outputs,
                           engine_core_timestamp=engine_core_timestamp, **kwargs)
        OutputProcessor.process_outputs = process_outputs

    def summarize(self):
        torch.cuda.synchronize()
        spans = {key: {'total_ms': sum(a.elapsed_time(b) for a, b in events),
                       'calls': len(events)} for key, events in self.events.items()}
        proposed = accepted = iterations = 0
        detail = []
        for p, tensor in self.acceptance:
            a = tensor.cpu().tolist()
            assert all(0 <= ai <= pi for ai, pi in zip(a, p))
            proposed += sum(p)
            accepted += sum(a)
            iterations += sum(pi > 0 for pi in p)
            detail.append({'proposed_per_request': p, 'accepted_per_request': a})
        selection = []
        for budget, valid, weight in self.selections:
            selection.append({'budget': budget.cpu().tolist(), 'valid': valid.cpu().tolist(),
                              'use_weight': weight})
        return {'stream_spans': spans, 'proposed_draft_tokens': proposed,
                'accepted_draft_tokens': accepted, 'request_verification_iterations': iterations,
                'batch_verification_iterations': sum(any(p) for p, _ in self.acceptance),
                'acceptance_rate': accepted / proposed if proposed else None,
                'accepted_tokens_per_verification': accepted / iterations if iterations else None,
                'verification_counters': detail, 'selection_samples': selection,
                'raw_qk_last_query_head0_first16': self.raw_score_probe.float().cpu().tolist()
                    if self.raw_score_probe is not None else None,
                'prefill_forward_ms_by_request': {
                    self.external_ids.get(rid, rid): sum(a.elapsed_time(b) for a, b in pairs)
                    for rid, pairs in self.prefill_events.items()}}


class MemoryMonitor:
    def __enter__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        # The benchmark restricts to physical GPU 0 in its run scripts.
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak = 0
        self.stop = threading.Event()
        def poll():
            while not self.stop.is_set():
                self.peak = max(self.peak, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
                self.stop.wait(0.01)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
        self.allocated = torch.cuda.max_memory_allocated()
        self.reserved = torch.cuda.max_memory_reserved()
