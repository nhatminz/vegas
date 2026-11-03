# Chuyển Vegas sang B200: chưa kiểm thử

Chưa có B200. Tài liệu này là kế hoạch triển khai, **không phải xác nhận tương thích**. Giữ nguyên `.venv` RTX 3090; clone/worktree khác và tạo `.venv-b200` mới. Không chép binary extension hoặc cache Triton/CUDA graph từ 3090.

## Stack và kiến trúc

| | RTX 3090 đã chạy | B200 cần xác minh |
|---|---|---|
| Compute capability | SM86, Ampere, 24 GiB GDDR6X | SM100, Blackwell, HBM3e; đọc dung lượng thực bằng NVML |
| Driver | 555.42.06 | Driver datacenter hỗ trợ B200 và CUDA được chọn; CUDA 12.8 GA đi cùng Linux >=570.26, nên chọn bản production mới được nhà cung cấp hỗ trợ |
| CUDA | Torch runtime 12.8; không có nvcc hệ thống | Toolkit >=12.8 để sinh native `sm_100`; driver/toolkit/runtime cần khớp |
| Python/Torch/Triton | 3.12.12 / 2.9.1+cu128 / 3.5.1 | Bắt đầu từ bộ version này làm ứng viên, kiểm tra SM100 trong Torch và Triton; chưa xác nhận chạy fork này trên B200 |
| vLLM | Vegas `7bf9d1edb8223307cff2089f855ca69cd094bc69`; Python editable, native ops từ parent `89a77b10846fd96273cce78d86d2556ea582d26e` | Rebuild hoặc wheel đúng ABI/commit có SM100; không thay Python Vegas bằng một vLLM mới rồi coi như giữ nguyên Vegas |

CUDA 12.8 sinh được cubin SM100. Cubin SM86/SM90 không phải binary dùng được trực tiếp trên SM100; PTX thông thường có thể forward JIT, nhưng PTX với tính năng riêng kiến trúc như `sm_90a` không phải giải pháp thay cho port. [NVIDIA Blackwell guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html), [CUDA 12.8 driver table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

## Modified FlashAttention của Vegas

`cmake/external_projects/vllm_flash_attn.cmake` pin [npz7yyk/vllm-flash-attn](https://github.com/npz7yyk/vllm-flash-attn/tree/0c05c4b59b4a795f2c2add264fef094e758478e3), commit `0c05c4b59b4a795f2c2add264fef094e758478e3`. Source tham khảo trên 3090 ở `.cache/vegas-flash-attn/`; **không build modified FA3 trên 3090**.

Các phần phụ thuộc cần đọc/port:

- `vllm_flash_attn/flash_attn_interface.py`: thêm `scores=(batch, q_heads, 2, max_len)`. Chỉ nhánh FA3 truyền buffer vào native op; nhánh FA2 không thu score. Dispatch FA3 của fork chỉ cho compute capability major 9, chặn major 10.
- `hopper/flash_api.cpp`, `hopper/flash.h`, `hopper/flash_fwd_launch_template.h`: truyền pointer/stride score, scheduler, paged KV và dispatch `ReturnLogits`.
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`: lưu raw QK **sau causal masking, trước softmax**, chỉ first/last query của từng request; PackGQA phải khôi phục đúng query head. Đây là mainloop dùng SM90 WGMMA/TMA, layout accumulator và warp-specialized pipeline; phải thiết kế lại hoặc tích hợp vào kernel Blackwell.
- Sparse draft cần **token-sized pages (`page_size=1`)**, page table riêng từng layer/request, budget khác nhau và recent slots. Không chỉ port đường verify liên tục.

FA4 hiện có backend tối ưu cho Blackwell, nhưng cài FA4 **không tự thêm** buffer `scores`, Collect-2-Query, layout/strides và page table của Vegas. Phải port giao diện lẫn kernel, rồi kiểm tra dispatch thực tế. [FlashAttention upstream](https://github.com/Dao-AILab/flash-attention).

Hai hướng triển khai: (1) thử portable reference backend dưới đây trên SM100, giữ overhead tính lại QK để đối chiếu correctness; hoặc (2) fuse collection vào backend Blackwell và thay sparse draft tương ứng. Hướng (2) phải giữ nguyên first/last theo request, raw score bf16, mean theo head/query, top-k theo budget, append recent KV và full-KV rejection verification. Không coi thay bằng streaming/sliding window là port Vegas.

## Patches 3090 và build mới

| File | Khi chuyển B200 |
|---|---|
| `.../attn_overrider/utils/sm86.py` | Giữ làm reference: Triton raw QK, reduction, sparse attention, hidden-state gather; torch top-k có mask/sort. Chưa benchmark SM100. Thay các primitive khi kernel fused đã đúng. |
| `.../attn_overrider/vegas.py` | Giữ `SCORE_MODE="logit"`, dispatch rõ và log backend. Portable bỏ nvcc/CUB workspace/autotune; native vẫn dùng code gốc. `auto` chỉ tự chọn portable cho SM8x. |
| `.../sparse_attn/proposer.py` | Giữ fix `arange(max_batch_size + 1)` và gather theo mapping. Portable gather dùng current stream; helper CUDA native gốc dùng default-stream launch, cần audit nếu bật async/nondefault stream. |
| `vllm/v1/sample/rejection_sampler.py` | Giữ observer optional và counters trong nhánh accept greedy/random; không thay acceptance/recovery probability. |
| `benchmarks/compare*.py`, `benchmark_compare.py` | Giữ definitions metrics/fairness; cập nhật backend option trước khi benchmark B200. |
| `scripts/run_compare.sh` | Bỏ ép SM86/portable/FA2 cho thí nghiệm B200. Harness hiện ép FA2 ở Python nữa: phải đổi đồng bộ và lưu cấu hình mới. |

Trên checkout B200, đặt `TORCH_CUDA_ARCH_LIST="10.0"` (hoặc `10.0+PTX` cho extension generic), CMake `CMAKE_CUDA_ARCHITECTURES=100`, hoặc `-gencode arch=compute_100,code=sm_100`. Kernel dùng tính năng `100a` cần target riêng đúng compiler/kernel; không đổi cơ học `90a` thành `100a`. Kiểm tra CMake lọc architecture, headers/CUTLASS/CUB và dtype hỗ trợ của từng extension. Các helper native `varlen_reduce.py`/`varlen_topk.py` hardcode fallback `8.0;8.9;9.0`: override env hoặc sửa cho SM100; autotune lại ngưỡng top-k vốn tối ưu H100.

Rebuild toàn bộ `.so` với Torch ABI đã chọn; cache extensions/Triton riêng. Kiểm tra `torch.cuda.get_device_capability()`, `torch.cuda.get_arch_list()`, đường import `vllm.__file__`, log backend, `cuobjdump --list-elf` cho SM100 và profiler cho kernel thực. FA3 `.so` import thành công không chứng minh dispatch/capability phù hợp. Chỉ thử `CUDA_FORCE_PTX_JIT=1` trong bài test riêng rồi unset; không dùng nó để che thiếu port API.

## Context, concurrency và bộ nhớ

Qwen3-4B local có 36 layer, 8 KV heads, head dim 128: BF16 KV khoảng **144 KiB/token/request**. Ước lượng KV đầy context, chưa gồm ~7.56 GiB weights, activation, graph, workspace và output/draft:

| Context | batch 8 | batch 16 | batch 32 |
|---|---:|---:|---:|
| 32K | 36 GiB | 72 GiB | 144 GiB |
| 64K | 72 GiB | 144 GiB | 288 GiB |
| 128K | 144 GiB | 288 GiB | 576 GiB |

Thử 32K: 8 → 16 → 32; 64K: 8 → 16 nếu còn bộ nhớ; 128K: bắt đầu 1/2/4, rồi 8 nếu fit. 128K×8 sát giới hạn; 64K×32 và 128K×16/32 không fit BF16 KV trên một B200 180GB. Muốn những tổ hợp đó cần GPU bổ sung hoặc đổi KV precision/offload và benchmark cả hai phương pháp trong cùng cấu hình. `max_model_len` là trần tổng prompt+output, không chứng minh prompt thật dài đến trần.

Local model card ghi native 32,768 và YaRN đến 131,072; `config.json` hiện `max_position_embeddings=40960`, chưa bật YaRN. Dùng model/config copy riêng, chẳng hạn `rope_scaling={"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}` cho bài test dài; baseline/Vegas phải dùng cùng config. Không sửa model đang dùng trên 3090. Proposer có TODO về RoPE khác: kiểm tra correctness với YaRN thật trước khi coi setting dài hợp lệ.

Bắt đầu `gamma=4`, ratio `0.07`, min KV `256`, BF16 và prefill chunks 4096/8192. Sau correctness, sweep gamma `{2,4,6,8}` × ratio `{0.03,0.05,0.07,0.10,0.15}` theo từng context/batch; ghi effective budget, acceptance và overhead. Chọn gamma theo throughput thực, không theo acceptance đơn lẻ. Tối ưu graphs/async sau eager reference; kiểm tra request churn, KV rollback và batch reorder lại.

## Checklist trên B200

- [ ] Môi trường mới, ghi driver/toolkit/Torch/Triton/vLLM/FA commit và native architecture.
- [ ] Baseline batch 1, rồi Vegas batch 1; xác nhận raw logits, sparse slots, draft và counter accept/reject.
- [ ] Chạy GPU reference tests; thêm SM100 fused-vs-portable test với varlen, GQA, masking, ties, budget khác nhau.
- [ ] Greedy differential và logprob probes; phân biệt sai thuật toán với sai số floating point, không hứa bitwise equivalence giữa kernel/batch shapes.
- [ ] Random sampling: full draft probabilities, residual recovery `max(p-q,0)`, top-k/top-p giống nhau, không penalty chưa hỗ trợ.
- [ ] Context thật 32K/64K/128K, YaRN, prompt+output+gamma margin; các batch fit bộ nhớ ở bảng trên.
- [ ] Warmup riêng; JSON/CSV, per-request timestamps, direct acceptance counters, profile riêng, NVML peak; không gọi thời gian toàn batch là latency/request.
- [ ] Chỉ tuyên bố tương thích sau khi inference, correctness và benchmark SM100 thành công.
