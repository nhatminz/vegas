# Chạy và kiểm chứng trên RTX 3090

Repository: `/mnt/hdd/nhatminh/SDLongContext/vegas`.
Model local: `/mnt/hdd/nhatminh/SpecDecode/models/Qwen3-4B`.
Dataset offline: `/mnt/hdd/nhatminh/data/{aime25,codeelo}`.

Kết quả tổng hợp và kiểm tra fairness ở [results/RESULTS.md](results/RESULTS.md), [CSV](results/comparison_summary.csv), [JSON](results/comparison_summary.json). Mỗi run có JSON đầy đủ, CSV tổng, CSV từng request và log tương ứng trong `results/logs/`. Chỉ `status=PASS` sau inference GPU và thu metrics thành công. Đây là performance benchmark, không chấm lời giải AIME/CodeElo.

## Môi trường đã cài

- Vegas commit `7bf9d1edb8223307cff2089f855ca69cd094bc69`, local changes chưa commit; mỗi JSON lưu hash các source thực thi, git status và upstream commit.
- Python 3.12.12, PyTorch 2.9.1+cu128, Triton 3.5.1, vLLM distribution `0.16.0+precompiled` (module `0.16.0`). Driver 555.42.06; `nvidia-smi` báo driver CUDA capability 12.5, Torch runtime là 12.8. **Không có system nvcc/toolkit**; đường portable không cần nvcc.
- Transformers 4.57.6, Datasets 5.0.1, FlashInfer 0.6.3. Native FA2 bundled trong vLLM, Python package version `2.7.2.post1`; không cài package `flash-attn` rời. Modified FA3 của Vegas không được build/dùng trên SM86.
- Native vLLM/FA2 extensions được lấy từ wheel đúng parent commit `89a77b10846fd96273cce78d86d2556ea582d26e`. Diff Vegas so với parent không sửa `csrc/`, nên giữ nguyên native ABI. Source Python Vegas được cài editable; baseline dùng cùng checkout và cùng extensions.
- Wheel `.cache/wheels/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl`, SHA256 `f066b2a2f8597a4a3ada8fbbfd122b59086864b2260ca42dc81bf9fb57af0c42`. [Nguồn wheel](https://wheels.vllm.ai/89a77b10846fd96273cce78d86d2556ea582d26e/vllm-0.16.0-cp38-abi3-manylinux_2_31_x86_64.whl).

Dependencies đã cài trong `.venv`, `pip check` không báo dependency hỏng. [requirements/rtx3090.lock.txt](requirements/rtx3090.lock.txt) pin packages, [results/environment_freeze.txt](results/environment_freeze.txt) ghi môi trường cài. Script tái tạo ở [scripts/setup_rtx3090.sh](scripts/setup_rtx3090.sh); không chạy nó để nâng cấp môi trường sang B200.

## Bốn lệnh đã chạy kiểm thử

Các lệnh này dùng model/dataset local, cache trên HDD, bật offline và tự `cd` vào repository. Defaults: BF16, greedy, seed 42, max len 4096, 64 output tokens, 1 warmup + 3 measurement rounds + 1 profile round riêng; gamma 4, ratio 0.07, min KV 256. `--output` là prefix; bỏ nó để tạo tên theo timestamp.

```bash
/mnt/hdd/nhatminh/SDLongContext/vegas/scripts/run_baseline.sh aime25 --batch-size 8 --output results/vllm_aime25_b8_l4096_n64
/mnt/hdd/nhatminh/SDLongContext/vegas/scripts/run_vegas.sh aime25 --batch-size 8 --output results/vegas_aime25_b8_l4096_n64
/mnt/hdd/nhatminh/SDLongContext/vegas/scripts/run_baseline.sh codeelo --batch-size 8 --output results/vllm_codeelo_b8_l4096_n64
/mnt/hdd/nhatminh/SDLongContext/vegas/scripts/run_vegas.sh codeelo --batch-size 8 --output results/vegas_codeelo_b8_l4096_n64
```

Đổi concurrency bằng `--batch-size 1|8|16`; đổi draft bằng `--gamma` và `--sparsity-ratio`. Cấu hình 8192/256 đã thử ở batch 8 và 16: thêm `--max-model-len 8192 --max-new-tokens 256`. Dùng `--dtype float16` để chạy FP16; probe token equality đã thực hiện cho FP16 batch 8/64 AIME25. Random sampling đã inference thành công với `--temperature 0.6 --top-p 0.95 --top-k 20` trên CodeElo batch 1.

Harness trực tiếp tương đương:

```bash
cd /mnt/hdd/nhatminh/SDLongContext/vegas
scripts/run_compare.sh --method vegas --dataset codeelo --batch-size 8 --gamma 4 --sparsity-ratio 0.07
```

`--method vllm` truyền **`speculative_config=None`**. `--method vegas` truyền method `sparse_attn`, algorithm `vegas`; `SCORE_MODE="logit"` được assert. Cả hai dùng FLASH_ATTN v2, eager/synchronous, prefix caching tắt, chunked prefill 4096 tokens, cùng sampling/dtype/max tokens. `ignore_eos=True` để mọi request tạo đúng số output token đã cấu hình. Không truncate prompt; bỏ các dòng vượt tổng prompt+output+gamma budget và ghi số dòng bị bỏ vào JSON. `--offset` chọn phần khác của dataset.

Mặc định có **tqdm**: thanh tổng các giai đoạn và thanh số **output token thực tế** cho từng Warmup, Measurement và Profile. Profile/warmup vẫn được tách khỏi throughput measurement. Tắt thanh bằng `--no-progress` trên **cả hai phương pháp** khi cần đo không có chi phí hiển thị; cấu hình này được ghi vào JSON và kiểm tra fairness.

Ma trận kết quả bàn giao ban đầu được đo trước khi bật tqdm; dùng `--no-progress` để tái tạo điều kiện hiển thị đó. Kiểm thử GPU sau cập nhật UI được lưu riêng ở `results/progress_baseline_codeelo_b8.*`, `results/progress_vegas_codeelo_b8.*` và `results/progress_ui_validation.json`, không ghi đè số liệu gốc.

Terminal mặc định in bảng dễ đọc. Baseline có acceptance/draft/KV selection là **N/A** vì không chạy speculative decoding; các trường tương ứng trong JSON vẫn là `null` đúng nghĩa không áp dụng, không phải inference thất bại. Dùng `run_vegas.sh` nếu muốn đo acceptance. Thêm `--print-json` để in lại JSON thô như phiên bản trước; JSON/CSV luôn được lưu dù chọn kiểu hiển thị nào. Engine và NCCL process groups được shutdown trong `finally`, kể cả khi benchmark lỗi.

## Dataset và prompt

`aime25/raw/test.jsonl`: 30 dòng, `problem:str`, `answer:int`, `id:str` (file gốc có answer số nguyên; không suy schema từ viewer). Revision `563bb8404243c5f09de6ec262f2db674fe5bce9b`. Prompt dùng problem và yêu cầu kết quả boxed.

`codeelo/raw/test.json`: 408 dòng. Prompt ghép title, description, input, output, interaction khi có, toàn bộ cặp ví dụ `[input, output]`, note, và yêu cầu C++17. Không dùng riêng description. Revision `45a2cbba03bd9d20fbca040c2f7da91b2594c444`.

Cả hai gọi `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` từ tokenizer local và truyền thẳng token IDs cho vLLM. `--no-thinking` là tùy chọn; không dùng trong số liệu mặc định. Mỗi dataset có `manifest.json`, bản raw và bản Arrow `save_to_disk` trong `arrow/`; tải lại bằng `scripts/prepare_offline_data.py` khi có mạng. Benchmark chỉ đọc raw local và không gọi Hub.

## Port SM86 và phần overhead

Mã gốc đã được kiểm tra ở proposer, runner `_sample`, rejection sampler, reducer/top-k, KV mapping và modified FA3. [sm86.py](vllm/v1/spec_decode/sparse_attn/attn_overrider/utils/sm86.py) cung cấp:

1. Triton tính lại **raw unscaled QK**, chỉ first/last query của từng request/head; causal mask và GQA đúng. Buffer score BF16 như fork gốc; không softmax để rank.
2. Triton mean reduction cùng quy tắc Collect-2-Query: verification lấy first+last, single-query lấy first, prefill lấy last.
3. `torch.topk` với validity mask, sort để mỗi request lấy đúng prefix budget riêng. Có thêm allocation/launch/sort so với CUB gốc; tie order không được cam kết.
4. Giữ nguyên Triton index-to-slot gốc, page table từng layer/request và append recent slots. Triton sparse GQA dùng online softmax và chỉ đọc các slot đang hợp lệ.
5. Triton gather padded/reordered hidden states, thay CUDA JIT helper khi batch/token counts lệch nhau. Giữ cách tính draft probabilities và full-KV verification của Vegas.

FA2 verify đầy đủ KV. Attention collection **tính lại hai QK rows**, không được fused miễn phí vào verify như modified Hopper FA3. Draft kernel portable chưa tối ưu tensor cores/split-K; top-k và Python eager có overhead. Không kỳ vọng bản này nhanh hơn baseline ở context ngắn. `VEGAS_BACKEND=portable` được scripts đặt rõ; `auto` chọn portable trên SM8x. Native đòi modified FA3/SM90/scores API và fail rõ nếu dispatch sai; không âm thầm dùng FA2 bỏ score.

Files gốc sửa: `attn_overrider/vegas.py`, `sparse_attn/proposer.py` (fix CPU query boundary + portable gather), `sample/rejection_sampler.py` (observer/counters optional trong nhánh accept, không sửa probability/recovery). Baseline không tạo attention overrider; mỗi benchmark process dùng một engine độc lập.

Chỉ kiểm thử vanilla causal GQA, BF16/FP16, single-query drafting, eager/sync, Qwen3-4B. CLI không cung cấp penalties/bad words/custom logits processors; không suy rộng lossless sampling sang các tính năng mà draft probability gốc chưa xử lý đầy đủ. Graphs/async, model khác và B200 chưa kiểm thử.

## Định nghĩa metric

| Metric | Nguồn / cách tính |
|---|---|
| E2E throughput | Tổng output tokens / tổng wall time `generate` của measurement; gồm scheduling, prefill/decode, không gồm load/tokenize/warmup |
| Decode-only throughput | Chỉ token emissions sau first token của request cuối cùng trong batch / thời gian từ mốc đó tới emission cuối; loại phần mixed prefill trước mốc |
| Acceptance | Counter trực tiếp trong nhánh accept greedy/random của rejection kernel / số draft được đưa vào verification. Trước scheduler truncation; không suy từ final output length. Baseline N/A |
| Accepted/verification | Accepted draft / số **request** verification iterations có draft; không cộng recovered/bonus. Batch iteration count lưu riêng |
| Prefill forward | CUDA event stream span target-forward khi request đang prefill; request chung batch chia sẻ service span, không chia cho batch size. Chunked prefill cộng các chunk |
| Prefill service | Native `first_token_ts - scheduled_ts`, loại queue nhưng gồm sampling và initial Vegas draft |
| TTFT | Native first-token latency tính từ request arrival; có queue/frontend overhead |
| TPOT | Từng request: `(last_token_ts - first_token_ts)/(output_tokens-1)`; token burst speculative được amortize, không phải đo ITL từng token trong burst |
| Draft/verification/KV selection | CUDA event spans từ **profile batch riêng**, tên `profile_*_ms_per_call`; verification forward gồm collection/reduction/top-k. Không cộng các span lồng nhau |
| Peak VRAM | NVML tổng device lấy mẫu 10 ms trong measurement; Torch allocated/reserved peaks lưu riêng. Gồm allocator/KV reservation/display |

Profile chi tiết được tách khỏi primary measurement để tránh per-layer CUDA events làm lệch throughput. Warmup chạy cùng shape/batch trước measurement; compile/autotune/model load không được tính vào inference throughput. JSON giữ từng verification count, selection budget/valid length, raw score probe, timestamps request, output IDs, config, versions và hashes.

## Correctness, tái chạy và giới hạn

```bash
cd /mnt/hdd/nhatminh/SDLongContext/vegas
PYTHONPATH="$PWD" TRITON_CACHE_DIR="$PWD/.cache/triton" .venv/bin/python -m pytest -q --confcutdir=benchmarks benchmarks/test_compare_kernels.py
.venv/bin/python scripts/run_validation_matrix.py base
.venv/bin/python scripts/run_validation_matrix.py long
.venv/bin/python scripts/summarize_compare.py
```

GPU tests dùng dense PyTorch reference: raw QK/masking/GQA và variable-length top-k đến 8191 keys, BF16/FP16, recent-slot mapping, sparse attention, reordered gather, greedy accept counts và stochastic recovered distribution. Log/JUnit lưu ở `results/`.

BF16 greedy **không bitwise giống baseline ở mọi request**. AIME25 batch 8/64 có 4/8 chuỗi khớp, còn lại target top-1/tie thay đổi; logprob evidence ở [greedy_precision_diagnostics.json](results/greedy_precision_diagnostics.json). FP16 probe cùng 8 prompt cho 8/8 chuỗi khớp. Điều này phù hợp với floating-point shape dependence nhưng không chứng minh mọi sai khác đều do số học. Rejection algorithm giữ full target verification, true draft distributions theo code gốc và residual `max(p-q,0)`; không tuyên bố bitwise losslessness mọi setting. Random output giữa AR/SD không cần giống theo cùng seed vì số lần lấy RNG khác nhau.

Các batch chính dùng prompt thật 49–850 tokens AIME25 và 265–1249 tokens CodeElo; không gọi trần 8192 là input 8192. Min KV 256 làm actual keep ratio cao hơn ratio 0.07 khi context ngắn. Chưa chấm task accuracy, chưa test context thật 32K+, chưa có B200. Xem [B200_MIGRATION_NOTES.md](B200_MIGRATION_NOTES.md) trước khi chuyển máy.
