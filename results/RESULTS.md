# Kết quả RTX 3090 thực đo

Mỗi ô cấu hình có 1 batch warmup, 3 batch measurement và 1 batch profile riêng. BF16, greedy, seed 42, gamma 4, ratio 0.07, min KV 256, eager, FA2; cùng prompts/token IDs và sampling giữa hai phương pháp.

**PASS ở đây xác nhận inference GPU và ghi metrics thành công. Không đồng nghĩa bitwise equivalence hoặc độ chính xác lời giải.** Prefill trong bảng là CUDA stream span target-forward theo request, không chia thời gian batch cho batch size. Prefill service, TTFT, queue và profile riêng nằm trong JSON/CSV.

## max_model_len=4096, max_new_tokens=64

| Dataset | Batch | Method | E2E tok/s | Decode tok/s | Accept % | Accepted/req verify | Prefill forward ms/req | TTFT ms/req | TPOT ms | Peak device MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aime25 | 8 | [vllm](vllm_aime25_b8_l4096_n64.json) | 419.86 | 458.83 | N/A | N/A | 117.39 | 120.00 | 17.436 | 19996.1 |
| aime25 | 8 | [vegas](vegas_aime25_b8_l4096_n64.json) | 335.61 | 372.86 | 98.10 | 3.924 | 119.87 | 172.93 | 20.078 | 19962.1 |
| codeelo | 8 | [vllm](vllm_codeelo_b8_l4096_n64.json) | 293.99 | 429.27 | N/A | N/A | 473.16 | 476.28 | 19.812 | 20320.1 |
| codeelo | 8 | [vegas](vegas_codeelo_b8_l4096_n64.json) | 256.40 | 360.80 | 96.70 | 3.868 | 477.68 | 540.21 | 20.718 | 20286.1 |
| aime25 | 16 | [vllm](vllm_aime25_b16_l4096_n64.json) | 679.63 | 872.78 | N/A | N/A | 346.85 | 350.14 | 18.332 | 20266.1 |
| aime25 | 16 | [vegas](vegas_aime25_b16_l4096_n64.json) | 601.18 | 781.24 | 99.05 | 3.962 | 351.82 | 411.71 | 19.197 | 20208.1 |
| codeelo | 16 | [vllm](vllm_codeelo_b16_l4096_n64.json) | 446.27 | 883.34 | N/A | N/A | 481.39 | 776.65 | 23.706 | 20374.1 |
| codeelo | 16 | [vegas](vegas_codeelo_b16_l4096_n64.json) | 391.98 | 732.29 | 97.16 | 3.886 | 487.66 | 883.75 | 25.814 | 20340.1 |

## max_model_len=8192, max_new_tokens=256

| Dataset | Batch | Method | E2E tok/s | Decode tok/s | Accept % | Accepted/req verify | Prefill forward ms/req | TTFT ms/req | TPOT ms | Peak device MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aime25 | 8 | [vllm](vllm_aime25_b8_l8192_n256.json) | 422.23 | 431.33 | N/A | N/A | 117.31 | 119.85 | 18.547 | 19996.1 |
| aime25 | 8 | [vegas](vegas_aime25_b8_l8192_n256.json) | 374.22 | 385.06 | 98.67 | 3.947 | 120.89 | 173.88 | 20.280 | 19966.1 |
| codeelo | 8 | [vllm](vllm_codeelo_b8_l8192_n256.json) | 382.50 | 426.45 | N/A | N/A | 476.08 | 479.22 | 19.050 | 20320.1 |
| codeelo | 8 | [vegas](vegas_codeelo_b8_l8192_n256.json) | 340.45 | 377.40 | 94.26 | 3.770 | 481.89 | 544.59 | 20.004 | 20270.1 |
| aime25 | 16 | [vllm](vllm_aime25_b16_l8192_n256.json) | 811.96 | 870.54 | N/A | N/A | 352.70 | 356.06 | 18.379 | 20266.1 |
| aime25 | 16 | [vegas](vegas_aime25_b16_l8192_n256.json) | 714.36 | 767.23 | 98.08 | 3.923 | 354.57 | 414.54 | 20.166 | 20224.1 |
| codeelo | 16 | [vllm](vllm_codeelo_b16_l8192_n256.json) | 694.32 | 860.83 | N/A | N/A | 484.17 | 781.57 | 19.971 | 20374.1 |
| codeelo | 16 | [vegas](vegas_codeelo_b16_l8192_n256.json) | 584.70 | 707.55 | 93.15 | 3.726 | 490.64 | 888.73 | 22.121 | 20356.1 |

## Kiểm tra so sánh

Mọi cặp đã được assert cùng model, prompt hash, dataset hash, sampling, dtype, seed, context/token limit, concurrency, warmup/rounds và versions. Acceptance lấy từ nhánh accept thực tế của rejection kernel, không lấy output length; bonus/recovered token không tính accepted draft.

| Dataset | Batch | Max len / new | Bitwise matched request outputs |
|---|---:|---:|---:|
| aime25 | 8 | 4096 / 64 | 12 / 24 |
| codeelo | 8 | 4096 / 64 | 21 / 24 |
| aime25 | 16 | 4096 / 64 | 36 / 48 |
| codeelo | 16 | 4096 / 64 | 36 / 48 |
| aime25 | 8 | 8192 / 256 | 12 / 24 |
| codeelo | 8 | 8192 / 256 | 9 / 24 |
| aime25 | 16 | 8192 / 256 | 21 / 48 |
| codeelo | 16 | 8192 / 256 | 3 / 48 |

BF16 không cho token equality ở mọi request. `greedy_precision_diagnostics.json` lưu first mismatch và target logprob margins; quan sát tie/top-1 đổi theo shape phù hợp với ảnh hưởng số học BF16, đây là giải thích dựa trên bằng chứng, không phải chứng minh cho mọi sai khác. Probe FP16 batch 8/64 trên cùng 8 bài AIME25 cho 8/8 token sequences giống nhau (`diagnostic_fp16_*.json`). Không tuyên bố bitwise losslessness cho mọi dtype/kernel/setting.

Giới hạn: prompts thật của các batch này dài 49–850 token (AIME25) và 265–1249 token (CodeElo). `max_model_len=8192` là trần cấu hình; đây không phải benchmark prompt 8K thật. Mức sàn 256 khiến effective keep ratio cao hơn 0.07 ở context ngắn; số selection sparse và khoảng ratio thực nằm trong summary CSV.

Peak VRAM là tổng device NVML lấy mẫu mỗi 10 ms trong measurement, có allocator/KV reservation và display. PyTorch allocated/reserved peak lưu riêng. Decode-only bắt đầu sau first-token của request cuối cùng; TPOT tính từng request, gồm thời gian chờ khi có mixed prefill/decode. Profile per-layer chạy ở batch riêng, không nằm trong throughput measurement.

Không OOM ở các cấu hình PASS liệt kê. Hai lỗi debug ban đầu đã được sửa: CUB workspace query và hidden-state gather cố gọi nvcc không tồn tại; giữ log trong `logs/*attempt01.log` và failure JSON ở `failures/`. B200 chưa được kiểm thử.
