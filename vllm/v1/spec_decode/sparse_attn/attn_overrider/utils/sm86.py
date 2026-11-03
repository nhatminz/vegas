"""Portable Vegas primitives. FA2 verifies; Triton collects QK and drafts.

This is Collect-2-Query, not an alternative KV policy: collect the first/last
query per request and head, average the same rows, select each request's top-k,
and append recent slots using Vegas' original metadata/index-to-slot kernel.
Unlike modified FA3, collection rereads K and recomputes two QK rows per layer.
Top-k uses sorted torch.topk with a validity mask (extra launches/allocation).
Only vanilla causal GQA, fp16/bf16, single-query drafting is supported here.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_hidden(SRC, CU, MAP, OUT, S0: tl.constexpr, S1: tl.constexpr,
                   H: tl.constexpr, HAS_MAP: tl.constexpr, BLOCK: tl.constexpr):
    b, token, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(CU + b - 1) if b > 0 else 0
    end = tl.load(CU + b)
    source = tl.load(MAP + b) if HAS_MAP else b
    hh = tile * BLOCK + tl.arange(0, BLOCK)
    mask = (token < end - start) & (hh < H)
    value = tl.load(SRC + source * S0 + token * S1 + hh, mask, 0)
    tl.store(OUT + (start + token) * H + hh, value, mask)


def gather_hidden(src, cu_num_draft_tokens, total_tokens, batch_size, reorder_mapping=None):
    """Exact copy of padded/reordered draft hidden states, on the current stream."""
    output = src.new_empty((total_tokens, src.shape[-1]))
    if total_tokens and batch_size:
        _gather_hidden[(batch_size, src.shape[1], triton.cdiv(src.shape[-1], 512))](
            src, cu_num_draft_tokens, reorder_mapping, output,
            src.stride(0), src.stride(1), src.shape[-1], reorder_mapping is not None,
            512, num_warps=4)
    return output


@triton.jit
def _collect(Q, K, CU, LENS, PT, S, QS0: tl.constexpr, QS1: tl.constexpr,
             KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
             PTS: tl.constexpr, SS0: tl.constexpr, SS1: tl.constexpr,
             SS2: tl.constexpr, PAGE: tl.constexpr, GROUP: tl.constexpr,
             D: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    b, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    first, end = tl.load(CU + b), tl.load(CU + b + 1)
    length = tl.load(LENS + b)
    kk = tile * BK + tl.arange(0, BK)
    dd = tl.arange(0, BD)
    pages = tl.load(PT + b * PTS + kk // PAGE, kk < length, 0)
    keys = tl.load(K + pages[:, None] * KS0 + (kk % PAGE)[:, None] * KS1
                   + (h // GROUP) * KS2 + dd[None, :],
                   (kk[:, None] < length) & (dd[None, :] < D), 0).to(tl.float32)
    q0 = tl.load(Q + first * QS0 + h * QS1 + dd, dd < D, 0).to(tl.float32)
    q1 = tl.load(Q + (end - 1) * QS0 + h * QS1 + dd, dd < D, 0).to(tl.float32)
    s0 = tl.sum(keys * q0[None, :], 1)
    s1 = tl.sum(keys * q1[None, :], 1)
    # Bottom-right aligned causal masking, as in FA3. Only valid keys enter
    # reduction; for prefill only the final query row is used.
    s0 = tl.where(kk <= length - (end - first), s0, float('-inf'))
    tl.store(S + b * SS0 + h * SS1 + kk, s0, kk < length)
    tl.store(S + b * SS0 + h * SS1 + SS2 + kk, s1, kk < length)


def collect_raw_scores(*, q, k, cu_seqlens_q, seqused_k, block_table,
                       max_seqlen_k, scores, **kwargs):
    _collect[(seqused_k.numel(), q.shape[1], triton.cdiv(max_seqlen_k, 64))](
        q, k, cu_seqlens_q, seqused_k, block_table, scores,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
        block_table.stride(0), scores.stride(0), scores.stride(1), scores.stride(2),
        k.shape[1], q.shape[1] // k.shape[2], q.shape[2],
        triton.next_power_of_2(q.shape[2]), 64, num_warps=4)


@triton.jit
def _reduce(X, VL, ENTRY, OUT, XS0: tl.constexpr, XS1: tl.constexpr,
            XS2: tl.constexpr, OS0: tl.constexpr, H: tl.constexpr,
            BH: tl.constexpr, BK: tl.constexpr):
    b, tile = tl.program_id(0), tl.program_id(1)
    kk, hh = tile * BK + tl.arange(0, BK), tl.arange(0, BH)
    length, entry = tl.load(VL + b), tl.load(ENTRY + b)
    off = b * XS0 + hh[:, None] * XS1 + kk[None, :]
    mask = (hh[:, None] < H) & (kk[None, :] < length)
    a = tl.load(X + off, mask, 0).to(tl.float32)
    z = tl.load(X + off + XS2, mask, 0).to(tl.float32)
    val = tl.where(entry == 1, a, tl.where(entry == 2, z, (a + z) * 0.5))
    val = tl.sum(val, 0) / H
    tl.store(OUT + b * OS0 + kk, val, kk < length)


def reduce_logit_scores(x, valid_lens, reduce_entry, output, **kwargs):
    _reduce[(x.shape[0], triton.cdiv(x.shape[-1], 64))](
        x, valid_lens, reduce_entry, output, *x.stride()[:3], output.stride(0),
        x.shape[1], triton.next_power_of_2(x.shape[1]), 64, num_warps=4)


def request_topk(metric, topks, valid_lens, output, **kwargs):
    # Sorting is essential: budgets differ by request, so each row must contain
    # its own top budget entries at the beginning, not an unsorted max-k set.
    mask = torch.arange(metric.shape[1], device=metric.device)[None, :] < valid_lens[:, None]
    ranked = metric.masked_fill(~mask, float('-inf'))
    width = min(output.shape[1], metric.shape[1])
    indices = torch.topk(ranked, width, dim=-1, sorted=True).indices
    output[:, :width].copy_(indices)


@triton.jit
def _sparse(Q, K, V, PT, LENS, CU, OUT, QS0: tl.constexpr,
            QS1: tl.constexpr, KS0: tl.constexpr, KS2: tl.constexpr,
            VS0: tl.constexpr, VS2: tl.constexpr, PTS: tl.constexpr,
            OS0: tl.constexpr, OS1: tl.constexpr, GROUP: tl.constexpr,
            D: tl.constexpr, BD: tl.constexpr, SCALE: tl.constexpr,
            BK: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    row, length = tl.load(CU + b), tl.load(LENS + b)
    dd, kk = tl.arange(0, BD), tl.arange(0, BK)
    q = tl.load(Q + row * QS0 + h * QS1 + dd, dd < D, 0).to(tl.float32)
    m = tl.full((), float('-inf'), tl.float32)
    l = tl.full((), 0., tl.float32)
    acc = tl.full((BD,), 0., tl.float32)
    for start in range(tl.cdiv(length, BK)):
        idx = start * BK + kk
        slots = tl.load(PT + b * PTS + idx, idx < length, 0)
        keys = tl.load(K + slots[:, None] * KS0 + (h // GROUP) * KS2 + dd[None, :],
                       (idx[:, None] < length) & (dd[None, :] < D), 0).to(tl.float32)
        vals = tl.load(V + slots[:, None] * VS0 + (h // GROUP) * VS2 + dd[None, :],
                       (idx[:, None] < length) & (dd[None, :] < D), 0).to(tl.float32)
        logits = tl.sum(keys * q[None, :], 1) * SCALE
        logits = tl.where(idx < length, logits, float('-inf'))
        nm = tl.maximum(m, tl.max(logits, 0))
        alpha = tl.exp(m - nm)
        p = tl.exp(logits - nm)
        acc = acc * alpha + tl.sum(p[:, None] * vals, 0)
        l = l * alpha + tl.sum(p, 0)
        m = nm
    tl.store(OUT + row * OS0 + h * OS1 + dd, acc / l, dd < D)


def sparse_attention(*, q, k, v, seqused_k, cu_seqlens_q, block_table,
                     max_seqlen_q, out=None, softmax_scale=None, **kwargs):
    if max_seqlen_q != 1 or k.shape[1] != 1:
        raise ValueError('Portable Vegas draft expects one query and token-sized pages')
    if any(kwargs.get(n) is not None for n in ('alibi_slopes', 's_aux', 'q_v')):
        raise NotImplementedError('Portable Vegas supports vanilla causal GQA only')
    if kwargs.get('softcap', 0) or kwargs.get('window_size') not in (None, [-1, -1], (-1, -1)):
        raise NotImplementedError('Portable Vegas does not implement softcap/local attention')
    if out is None:
        out = torch.empty_like(q)
    _sparse[(seqused_k.numel(), q.shape[1])](
        q, k, v, block_table, seqused_k, cu_seqlens_q, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(2),
        v.stride(0), v.stride(2), block_table.stride(0),
        out.stride(0), out.stride(1), q.shape[1] // k.shape[2], q.shape[2],
        triton.next_power_of_2(q.shape[2]), softmax_scale or q.shape[2] ** -0.5,
        32, num_warps=4)
    return out
