# Fork notes: CUDA on RTX 4060 Ti 16 GB (Ada, sm_89), Fedora VM on Proxmox

Branch `cuda-ada-4060ti`, based on [PrismML's `prism`](https://github.com/PrismML-Eng/llama.cpp) (the same base as
`vk-rdna1-5600m`, which holds the separate MoltenVK/Radeon work). Nothing from that branch is carried here.

## Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB, sm_89 |
| CPU / RAM | Ryzen 7 3700X (8 cores), 60 GB |
| OS | Fedora 44 **virtual machine** (QEMU/KVM) on **Proxmox** |
| Host toolchain | CUDA 12.6; the host nvcc cannot compile against GCC 16, so builds run in a CUDA container |

## The VM is part of the measurement

Proxmox has a measurable effect on results from this machine. In particular, **KSM (kernel same-page merging) on the
Proxmox host hurt performance.** Numbers from this branch are for this VM configuration and may differ on bare metal.

- KSM is a host setting. The guest cannot see or change it (`/sys/kernel/mm/ksm/run` reads 0 inside the VM).
- When recording a benchmark, note the KSM state of the host and whether the guest's memory is ballooned or pinned.
- Compare runs only under the same host configuration.

## Changes

None yet. Candidates are tracked in the hub repo under `patches/llama.cpp/` (MoE expert arena and KV streaming,
MTP draft arena, IPv6 RPC transport).

## Licence

llama.cpp and ggml are MIT. No model weights are included.
