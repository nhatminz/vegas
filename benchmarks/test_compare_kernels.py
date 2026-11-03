"""GPU correctness checks against independent dense PyTorch references."""
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils.sm86 import (
    collect_raw_scores, reduce_logit_scores, request_topk, sparse_attention, gather_hidden,
)
from vllm.v1.spec_decode.sparse_attn.attn_overrider.vegas import _index_to_slot_kernel
from vllm.v1.sample.rejection_sampler import rejection_sample

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU required')


@pytest.mark.parametrize('reorder', [True, False])
def test_gather_variable_drafts_and_reordered_requests(reorder):
    src = torch.arange(4 * 4 * 2560, device='cuda').reshape(4, 4, 2560).to(torch.bfloat16)
    counts = [4, 0, 2, 1]
    cu = torch.tensor(counts, device='cuda', dtype=torch.int32).cumsum(0).int()
    mapping = torch.tensor([2, 0, 3, 1], device='cuda', dtype=torch.int32) if reorder else None
    result = gather_hidden(src, cu, sum(counts), 4, mapping)
    ref = torch.cat([src[mapping[i].item() if reorder else i, :n] for i, n in enumerate(counts)])
    torch.testing.assert_close(result, ref, atol=0, rtol=0)


@pytest.mark.parametrize('lengths', [(513, 79, 19), (8191, 2317, 19)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_collect_reduce_request_topk(lengths, dtype):
    torch.manual_seed(13)
    device, qlens = 'cuda', [5, 17, 1]
    heads, kvheads, dim, page = 32, 8, 128, 16
    pages = (max(lengths) + page - 1) // page
    maxlen = pages * page
    table = torch.randperm(3 * pages, device=device).reshape(3, pages).int()
    k = torch.randn(3 * pages, page, kvheads, dim, device=device, dtype=dtype)
    q = torch.randn(sum(qlens), heads, dim, device=device, dtype=dtype)
    cu = torch.tensor([0, 5, 22, 23], device=device, dtype=torch.int32)
    lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    scores = torch.full((3, heads, 2, maxlen), float('nan'), device=device, dtype=torch.bfloat16)
    collect_raw_scores(q=q, k=k, cu_seqlens_q=cu, seqused_k=lens,
                       block_table=table, max_seqlen_k=max(lengths), scores=scores)
    refs = []
    for b in range(3):
        keys = k[table[b].long()].reshape(-1, kvheads, dim)[:lengths[b]]
        expanded = keys.repeat_interleave(heads // kvheads, dim=1).float()
        rows = q[cu[b].item():cu[b+1].item()][[0, -1]].float()
        ref = torch.einsum('qhd,khd->hqk', rows, expanded)
        ref[:, 0, lengths[b]-qlens[b]+1:] = float('-inf')
        refs.append(ref.to(scores.dtype))
        torch.testing.assert_close(scores[b, :, :, :lengths[b]], refs[-1], atol=0.125, rtol=0.008)
    valid = torch.tensor([lengths[0]-4, lengths[1], lengths[2]], device=device, dtype=torch.int32)
    entry = torch.tensor([0, 2, 1], device=device, dtype=torch.int32)
    reduced = torch.empty(3, maxlen, device=device, dtype=scores.dtype)
    reduce_logit_scores(scores, valid, entry, reduced)
    for b in range(3):
        n = valid[b].item()
        # Reduction reference uses the collected bf16 values (as native FA3 does).
        values = scores[b, :, :, :n].float()
        ref = values.mean((0, 1)) if b == 0 else values[:, 1 if b == 1 else 0].mean(0)
        torch.testing.assert_close(reduced[b, :n], ref.to(scores.dtype), atol=0.016, rtol=0.008)
    budgets = torch.tensor([11, 7, 19], device=device, dtype=torch.int32)
    out = torch.empty(3, 32, device=device, dtype=torch.int32)
    request_topk(reduced, budgets, valid, out)
    for b in range(3):
        n = budgets[b].item()
        chosen = out[b, :n].long()
        assert (chosen < valid[b]).all()
        expected = reduced[b, :valid[b].item()].topk(n).values.sort().values
        torch.testing.assert_close(reduced[b, chosen].sort().values, expected)


def test_index_to_slot_recent_slots_and_per_request_budgets():
    table = torch.tensor([[5, 1, 3], [0, 6, 2]], device='cuda', dtype=torch.int32)
    budget = torch.tensor([2, 3], device='cuda', dtype=torch.int32)
    valid = torch.tensor([17, 10], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([21, 14], device='cuda', dtype=torch.int32)
    indices = torch.zeros(2, 2, 16, device='cuda', dtype=torch.int32)
    indices[:, 0, :2] = torch.tensor([0, 16], device='cuda')
    indices[:, 1, :3] = torch.tensor([1, 5, 9], device='cuda')
    logical = indices.clone()
    _index_to_slot_kernel[(2, 2)](indices, table, budget, valid, lengths,
                                indices.stride(0), indices.stride(1), table.stride(0),
                                16, BLOCK_SIZE=256)
    for layer in range(2):
        for b in range(2):
            n = budget[b].item()
            selected = logical[layer, b, :n]
            recent = torch.arange(valid[b].item(), lengths[b].item(), device='cuda')
            all_indices = torch.cat([selected, recent]).long()
            expected = table[b, all_indices // 16] * 16 + all_indices % 16
            torch.testing.assert_close(indices[layer, b, :len(expected)].long(), expected.long())


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_sparse_attention_matches_dense_selected_kv(dtype):
    torch.manual_seed(27)
    q = torch.randn(3, 32, 128, device='cuda', dtype=dtype)
    k = torch.randn(128, 1, 8, 128, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.stack([torch.randperm(128, device='cuda')[:64] for _ in range(3)]).int()
    lens = torch.tensor([43, 17, 1], device='cuda', dtype=torch.int32)
    cu = torch.arange(4, device='cuda', dtype=torch.int32)
    actual = sparse_attention(q=q, k=k, v=v, seqused_k=lens, cu_seqlens_q=cu,
                              block_table=table, max_seqlen_q=1)
    for b in range(3):
        ids = table[b, :lens[b].item()].long()
        keys = k[ids, 0].repeat_interleave(4, dim=1).float()
        values = v[ids, 0].repeat_interleave(4, dim=1).float()
        logits = torch.einsum('hd,khd->hk', q[b].float(), keys) / 128**0.5
        ref = torch.einsum('hk,khd->hd', logits.softmax(-1), values).to(q.dtype)
        torch.testing.assert_close(actual[b], ref, atol=0.016, rtol=0.016)


def test_rejection_kernel_counts_greedy_accept_branches():
    ids = torch.tensor([1, 2, 3, 0, 2, 1], device='cuda', dtype=torch.int32)
    argmax = torch.tensor([1, 2, 0, 0, 2, 1], device='cuda')
    logits = torch.full((6, 4), -20., device='cuda')
    logits.scatter_(1, argmax[:, None], 20.)
    counts = torch.empty(3, device='cuda', dtype=torch.int32)
    output = rejection_sample(ids, [4, 2, 0], 4,
        torch.tensor([4, 6, 6], device='cuda', dtype=torch.int32), None, logits,
        torch.tensor([[3], [0], [1]], device='cuda', dtype=torch.int32),
        SimpleNamespace(all_greedy=True, all_random=False), accepted_counts=counts)
    assert counts.tolist() == [2, 2, 0]
    assert output.tolist() == [[1, 2, 0, -1, -1], [2, 1, 0, -1, -1], [1, -1, -1, -1, -1]]


def test_stochastic_rejection_recovers_target_distribution_and_counts():
    torch.manual_seed(314)
    n = 8192
    draft = torch.tensor([0.2, 0.8], device='cuda').expand(n, -1).contiguous()
    target = torch.tensor([0.8, 0.2], device='cuda').expand(n, -1).contiguous()
    ids = torch.multinomial(draft, 1).flatten().int()
    counts = torch.empty(n, device='cuda', dtype=torch.int32)
    sampling = SimpleNamespace(all_greedy=False, all_random=True, generators={},
                                temperature=torch.ones(n, device='cuda'))
    out = rejection_sample(ids, [1] * n, 1, torch.arange(1, n+1, device='cuda', dtype=torch.int32),
        draft, target.log(), torch.zeros(n, 1, device='cuda', dtype=torch.int32),
        sampling, accepted_counts=counts)
    # The target distribution and theoretical mean acceptance min(p,q).sum().
    assert abs((out[:, 0] == 0).float().mean().item() - 0.8) < 0.02
    assert abs(counts.float().mean().item() - 0.4) < 0.025
    assert ((out[:, 1] >= 0) == counts.bool()).all()
