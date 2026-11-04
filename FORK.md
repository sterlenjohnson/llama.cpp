# Fork notes: Vulkan on AMD under MoltenVK (Radeon Pro 5600M)

[PrismML's `prism`](https://github.com/PrismML-Eng/llama.cpp) plus 12 commits that make the Vulkan backend fast and
correct on a 2019 16" Intel MacBook Pro (Radeon Pro 5600M, RDNA1, 8 GB HBM2) through MoltenVK. Metal is broken on this
discrete GPU, so Vulkan is the only fast path. Companion measurements and negative results: engine repo (link to be added).

## Highlights

| Change | Effect |
|---|---|
| Emulated integer-dot matmul on by default (AMD + MoltenVK) | OLMoE prefill **120 → 245** tok/s (2.0x) |
| MoE `mul_mat_id` tile chosen by rows per expert | LFM2.5-24B IQ2_XXS prefill **56 → 165** tok/s (2.9x) |
| 4 rows/workgroup for K-quant mat-vec | Ornith 9B Q4_K_M decode 28.0 → 29.9 tok/s (+6%) |
| Subgroup arithmetic on (MoltenVK >= 1.4.2) | OLMoE decode 115 → 121, LFM2.5-24B 35.3 → 36.9 tok/s (+4-5%) |
| Multi-row decode attention kernels | OLMoE decode at 4096 ctx **46 → 94** tok/s; Qwen3-4B at 8192 ctx 27.0 → 39.5 (+46%) |
| `gated_delta_net` and external-memory-host fixes | correctness for Qwen3.5-style models; host-memory import on MoltenVK >= 1.4.1 |

Prefill is `-p 512` (default batch), decode `-n` tokens, `-ngl 99 -fa 0`, model fully in VRAM, GPU cooled between runs.
Decode barely moves at short context (bandwidth-bound); the large wins are prefill and long-context decode.

Caveats: these figures come from the measurement record on the pre-rebase base. After rebasing onto the current `prism`
I re-ran only Ornith 9B Q4_K_M (prefill 50.8, decode 29.6 tok/s, within noise of the earlier build). Single machine;
numbers say nothing about other GPUs.

## Environment switches

| Variable | Effect |
|---|---|
| `GGML_VK_DISABLE_INT_DOT` / `GGML_VK_FORCE_INT_DOT` | integer dot off / forced on |
| `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ` | rows per workgroup for K-quants / standard quants (clamped 1-8) |
| `GGML_VK_MOE_TILE_BY_BATCH`, `GGML_VK_MOE_TILE_ROWS_PER_EXPERT` | old / new MoE tile selection |
| `GGML_VK_DISABLE_AMD_MOLTENVK_SUBGROUP_ARITH` | subgroup arithmetic off |
| `GGML_VK_DISABLE_P021_MR`, `GGML_VK_DISABLE_NC_MR`, `GGML_VK_P021_SPLIT` | opt out of / tune the multi-row attention kernels |
| `GGML_VK_FORCE_ARCHITECTURE`, `GGML_VK_SHADER_CORE_COUNT` | override architecture and core-count detection |

## Verification

On the current `prism` tip the Vulkan `MUL_MAT` operator suite passes (3027 of 3027) and a deterministic generation on a
9B Q4_K_M model is correct. Not re-run after the rebase: perplexity, and other models. An earlier, broader subgroup change
corrupted Llama 3 and Gemma output, so compare your model's output or perplexity before trusting a speedup.

## Licence

llama.cpp and ggml are MIT. PrismML's `ggml/src/ggml-sycl/ternsycl/` is BSD-3-Clause (Intel) and untouched here.
No model weights are included.
