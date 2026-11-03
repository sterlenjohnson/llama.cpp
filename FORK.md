# Fork notes: Vulkan on AMD under MoltenVK (Radeon Pro 5600M)

This branch is [PrismML's `prism` branch](https://github.com/PrismML-Eng/llama.cpp) plus 12 commits that make
the Vulkan backend fast and correct on one machine: a 2019 16-inch Intel MacBook Pro with a Radeon Pro 5600M
(RDNA1 / Navi 12, 8 GB HBM2) driven through MoltenVK, because Metal is broken on that discrete GPU.

Everything here was measured on that one machine. Treat the numbers as evidence about this GPU, not about
GPUs in general. The measurements, patch rationale and negative results live in the companion engine repository
(link to be added).

## What the commits do

- MoltenVK subgroup workaround for `gated_delta_net` (Qwen3.5-style models).
- MoltenVK external-memory-host gate by version (>= 1.4.1) instead of a blanket disable, plus AMD
  architecture and shader-core-count overrides.
- Emulated integer dot enabled by default on AMD + MoltenVK; 4 rows per workgroup for K-quant mat-vec.
- MoE `mul_mat_id` tiles chosen by rows per expert on AMD + MoltenVK.
- Subgroup arithmetic enabled by default on AMD + MoltenVK >= 1.4.2.
- Multi-row decode attention kernels (`mul_mat_vec_p021_mr`, `mul_mat_vec_nc_mr`) for long-context decode.

## Environment switches added

| Variable | Effect |
|---|---|
| `GGML_VK_DISABLE_INT_DOT` / `GGML_VK_FORCE_INT_DOT` | turn the emulated integer dot off / force it on |
| `GGML_VK_RM_KQ`, `GGML_VK_RM_STDQ` | rows per workgroup for K-quants / standard quants (clamped 1-8) |
| `GGML_VK_MOE_TILE_BY_BATCH`, `GGML_VK_MOE_TILE_ROWS_PER_EXPERT` | old / new MoE tile selection |
| `GGML_VK_DISABLE_AMD_MOLTENVK_SUBGROUP_ARITH` | force subgroup arithmetic off again |
| `GGML_VK_DISABLE_P021_MR`, `GGML_VK_DISABLE_NC_MR`, `GGML_VK_P021_SPLIT` | opt out of / tune the multi-row attention kernels |
| `GGML_VK_FORCE_ARCHITECTURE`, `GGML_VK_SHADER_CORE_COUNT` | override architecture detection and core count |

## Verification

Rebuilt on the current `prism` tip: the `MUL_MAT` operator suite on the Vulkan device passes (3027 of 3027) and a
deterministic generation on a 9B Q4_K_M model is correct. Not re-run after the rebase: perplexity, and models
other than the one 9B model. An earlier, broader subgroup change corrupted Llama 3 and Gemma output, so check
your model with a perplexity or output comparison before trusting a speedup.

## Licence

llama.cpp and ggml are MIT. PrismML's `ggml/src/ggml-sycl/ternsycl/` is BSD-3-Clause (Intel); this branch does
not touch it. No model weights are included.
