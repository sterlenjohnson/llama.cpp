# Fork notes: Vulkan on AMD under Linux (Radeon Pro 5600M, T2 MacBook Pro, Fedora)

The same 2019 16" MacBook Pro and Radeon Pro 5600M (RDNA1, 8 GB HBM2) as the sibling branch
[`vk-rdna1-5600m`](https://github.com/sterlenjohnson/llama.cpp/blob/vk-rdna1-5600m/FORK.md), but booted into Linux with
the T2 kernel, so the GPU runs on native Mesa RADV instead of MoltenVK. This branch is **upstream `ggml-org/llama.cpp`
at `c7fc50941` plus 13 commits** (not PrismML's `prism`): it does not carry the Prism low-bit formats, and nothing from
the MoltenVK branch is in it.

## Test machine

| | |
|---|---|
| GPU | Radeon Pro 5600M (Navi 12, gfx1011), 8 GB HBM2, PCIe 3.0 x8 (measured 6.8 GB/s host to device) |
| CPU / RAM | Core i9-9880H (8 cores), 16 GB |
| OS | Fedora 44, kernel `7.2.8-310.t2.fc44`, Mesa 26.2.3 RADV, Vulkan 1.4.354 |

## Platform

Fedora 44 on the t2linux kernel: install notes at the [t2linux Fedora guide](https://wiki.t2linux.org/distributions/fedora/installation/),
kernel packages from the [`t2linux/fedora-kernel`](https://github.com/t2linux/fedora-kernel) project (COPR `sharpenedblade/t2linux`).

## Highlights

| Result | Effect |
|---|---|
| Linux against macOS, identical flags | prefill **3.7 to 31x**, decode **1.2 to 2.7x** |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` (without it the model sits in system RAM) | OLMoE decode **8.03 to 171** tok/s |
| p021 heads-first dispatch (the long-context crash) | OLMoE decode at 16k **17.8 to 54.2** tok/s, byte-identical output |
| RDNA1 decode kernels (Q1_0 int-dot, lane-per-block Q2_0/Q4_0, FA split_k) | Bonsai 8B 44.8 to 106.4, Bonsai 27B 21.0 to 28.8, Gemma 4 12B 25.7 to 34.8 tok/s |
| MoE expert cache on a 35B-A3B, 128k context, q4_0 KV, `--n-cpu-moe 26` | decode **17.2 to 27.2** tok/s (+58%) |
| 5600M as an RPC worker for a 4060 Ti, 27B Q5 split | decode **11.3 to 16.5** tok/s, 13k-token prefill **11.7 to 183** tok/s (macOS to Linux node) |
| Speculation (MTP / external draft) | a loss on the 35B (-21 to -58%); a win only when the model fits in VRAM (+28% on Qwen3.5-4B) |

## RPC boost with the 4060 Ti

The 5600M also works as the node behind a 4060 Ti 16 GB hub over RPC (27B Q5, 100k context, split 36/64, MTP draft,
wired IPv6). Swapping only the node from macOS/MoltenVK to this Linux/RADV build, decode goes from 11.3 to 16.5 tok/s
(a step of 0.154 s down to 0.109 s) and long-prompt prefill from 11.7 to 183 tok/s. The split is node-bound on both
systems: during the 13k prefill the 5600M is 87% busy on average while the 4060 Ti sits at 12%. The dual-stack (IPv6)
RPC transport is in this branch; the hub must run the same patch.

## Power and thermal unlock

The 5600M and the CPU share one heat pipe, and the T2's SMC sheds heat by cutting the GPU first, then the CPU. The scripts
in [`scripts/t2-5600m/`](scripts/t2-5600m/README.md) are what this branch's numbers were taken with.

- **The decode clamp is a hardware assertion.** The 35B's slow mode (about 11 tok/s) is every core pinned at 800 MHz while the
  GPU falls from 1145 to 500 MHz, with the package at only 59 to 65 C. Clearing MSR `0x1FC` bit 0 (BD PROCHOT) with `prochot.py`
  lifts it: the clamp costs about 5% of decode at 32k and about 30% at 128k (11.7 against 16.5 to 17.5 tok/s). A reboot re-arms it.
- **Follow the SMC, don't fix the cap.** A fixed 40 W GPU cap was killed by the watchdog at 101 C, 63% through the 128k prefill.
  Following the SMC's own request (`smc_cap_follow.py`) finished it: 130.6 tok/s prefill, junction steady at 99 C, no root.
- **Caps:** 35 W is the one that finishes at every depth (91 C at 128k); 40 W reaches 102 C at 128k with no margin; at 45 W
  runs died at 101 to 104 C. The 2k row is the control: the cap does not bind there.
- **Caveats:** `power_cap_sweep.sh` is untested here, raising the cap above its default needs the amdgpu overdrive mask, and
  clearing BD PROCHOT removes a safety net, so watch the junction temperature. The scripts are for this model of Mac only.

## Caveats

- Single runs on this laptop swing 1.5 to 1.7x; only effects above about 10% are quoted as real.
- The RPC comparison also moves the hub runtime by five weeks of upstream (ggml 0.24.0 to 0.27.0), and the original
  prompt text is lost, so acceptance varies with content and the short-prompt column is not like for like.
- The branch is 13 commits; the portable patch series in the engine repo covers the first 10 (byte-identical), and the
  last three (MMVQ rows switch, PCIe byte accounting, a pipeline-parallel dead end) are newer.
- The 35B speed depends on the laptop's SMC: an 800 MHz CPU clamp (likely PROCHOT) drops decode to about 11 tok/s.

## Environment switches

| Variable | Effect |
|---|---|
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | **required**: keep the model in VRAM, not GTT |
| `GGML_VK_DISABLE_P021_MR` | back to the old p021 dispatch order |
| `GGML_VK_DISABLE_Q1_0_MMVQ`, `_Q2_0_MMVQ_T`, `_Q4_0_MMVQ_T`, `_FA_QSPLIT` | turn off an RDNA1 decode kernel |
| `GGML_VK_DISABLE_MMVQ` | turn off the IQ2_S/IQ3_S int8 mat-vecs |
| `GGML_VK_DISABLE_FA_WIDE`, `_RS1`, `_DS4`, `_FA_GQA` | turn off the FA wide loads and GQA tile staging |
| `--moe-cache-profile`, `--moe-cache-profile-save`, `GGML_MOE_HOT_FRACTION` | MoE cache expert profile |
| `LLAMA_MOE_CACHE_SYNC_LEGACY=1` | three waits per cached layer again |

## Recommended 35B-A3B config (128k)

```
-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -c 131072 --n-cpu-moe 26 --load-mode none --moe-cache-mib 1280 -ub 256   # decode, ~27 tok/s
-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -c 131072 --n-cpu-moe 26 --load-mode none --moe-cache-mib 1024 -ub 512   # prompts, ~26 tok/s
```

## Full record

Every measurement, the scripts and the rejected ideas are in the engine repo's `engine/linux/` (README,
`ORDER_OF_EVENTS.md`, `patches/`). Before trusting any speedup, compare output or perplexity against an unpatched build;
`ppl_check.sh` does this against the macOS runs.

## Licence

llama.cpp and ggml are MIT. No model weights are included.
