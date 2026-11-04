# Fork notes: CUDA on RTX 4060 Ti 16 GB (Ada, sm_89), Fedora VM on Proxmox

Branch `cuda-ada-4060ti`, based on [PrismML's `prism`](https://github.com/PrismML-Eng/llama.cpp). The sibling branch
`vk-rdna1-5600m` holds the separate MoltenVK/Radeon work for a 2019 MacBook Pro; nothing from it is carried here.

**Status: no code changes yet.** This branch currently equals `prism` plus this file. It exists so the 4060 Ti work has
a home, a stated test machine, and a record of what has already been measured. The candidate changes and their
measurements are below; none of them is applied on this branch.

## Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB, sm_89 |
| CPU / RAM | Ryzen 7 3700X (8 cores), 60 GB |
| OS | Fedora 44 **virtual machine** (QEMU/KVM) on **Proxmox** |
| Toolchain | CUDA 12.6 on the host; the host nvcc cannot compile against GCC 16, so builds run in a CUDA container (`-j2`, 40 GB memory cap) |

## The VM is part of the measurement

Proxmox has a measurable effect on results from this machine. **KSM (kernel same-page merging) on the Proxmox host
hurt performance.** KSM is a host setting: the guest cannot see or change it (`/sys/kernel/mm/ksm/run` reads 0 inside
the VM). Record the host's KSM state with every benchmark and compare runs only under the same host configuration.
Numbers here may differ on bare metal.

## Candidate changes (not applied here)

These live in the llamaHub repo under `patches/llama.cpp/`. They were made on a hand-merged tree
(`moe-cache` + `adaptive-kv-streaming`), not on `prism`, so each needs a rebase and a re-measurement before it counts as
a change of this branch.

| Patch set | What it does |
|---|---|
| `moecache-kv-stream/` (8 patches) | MoE expert cache (`--moe-expert-cache-size`) and adaptive KV streaming arena (`--kv-stream-arena-mib`), with merge fixes |
| `kv-stream-mtp-draft-arena.patch` | keeps the MTP draft cache off the KV arena |
| `rpc-ipv6-dualstack.patch`, `rpc-ipv6-getaddrinfo.patch` | IPv6 and dual-stack transport for the RPC backend |

Constraints of the arena, enforced by the code: it loads only `LLM_ARCH_QWEN35`, it refuses draft/MTP contexts (MTP and
the arena are exclusive), and the compact causal mask and streamed attention are exclusive.

## Benchmarks

Prefill / decode in tok/s. Every row names the tree it was measured on.

**Hand-merged tree, 4060 Ti, Qwen3.8 Flash Next, 192k context, identical flags, 57,601-token prompt**

| Build | Prefill | Decode |
|---|---|---|
| GS `qwen3next-moecache-gs-cuda-rpc` | 209.0 | 10.14 |
| moecache + KV-stream merge | 243.1 | 8.02 |

The merge trades decode for prefill on the expert-cache path. MTP speculation ran at full speed (23.68 tok/s at 131k
context), and the expert cache costs about 21% decode against the GS build.

**Hand-merged tree, Qwen3.8-27B-GSQ-RCO-IQ3_XXS, 6,196-token prompt (prefill / decode)**

| Build | No arena | Arena 2 GiB |
|---|---|---|
| standalone adaptive | 652.7 / 21.95 | 646.2 / 21.14 |
| moecache + KV-stream merge | 663.6 / 21.40 | 309.3 / 14.71 |

Open problem: the arena halves prefill on the merge. Ruled out by measurement: auto-fit, arena size (4 GiB recovers
decode to 18.66), and a silent fall back out of the streamed kernels. The gap widens with context (2x at 6k, about 10x at
57.6k), which points at resident-page churn.

**Strata engine on the same box (not llama.cpp), Swift Qwen3.8 Flash-Next IQ3_XXS, 262k context, 300 tokens, temperature 0**

Included because it is the comparison this branch is trying to close. Three short prompts, one run each, so treat it as
indicative only. DFlash2 ran on upstream Strata v0.2.10 without `--pool-workers`; MTP ran on the older production binary
with its full config.

| Drafter | Decode | Drafts accepted |
|---|---|---|
| MTP (production) | 36.2 / 43.0 / 38.5 | about 78% |
| DFlash2 | 27.1 / 28.0 / 28.1 | about 66% |

For scale, an earlier same-box run measured stock llama.cpp CPU-offloaded MoE at about 1.7 tok/s decode against Strata's
22 to 26 (IQ2_XS vs IQ4_XS quants, different context lengths), so the engine, not the quant, dominates.

## Verification

None of the candidate patches has been rebuilt or run on `prism` yet. Before trusting any speedup, compare output or
perplexity against the unpatched build.

## Licence

llama.cpp and ggml are MIT. No model weights are included.
