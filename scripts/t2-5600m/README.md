# Power and thermal unlock scripts (MacBookPro16,4, Radeon Pro 5600M, Linux)

Copied from the engine repo's `engine/linux/`, where the measurements behind them live. Tested on a T2 MacBook Pro 16"
(i9-9880H, Radeon Pro 5600M) running Fedora 44 with the t2linux kernel. They write to hardware: read them before running.

| Script | What it does | Needs |
|---|---|---|
| `prochot.py` | `status` / `off` / `on` / `watch`: reads and clears MSR `0x1FC` bit 0 (BD PROCHOT) on every CPU, saves the original values to `/tmp/prochot-1fc.saved` and restores them on `on`. Volatile: a reboot re-arms it | root, `modprobe msr` |
| `smc_cap_follow.py` | follows the SMC's own dGPU power request (`smc_plimit_raw`) and writes it back as a GPU cap through LACT's socket | LACT, t2 SMC driver; no root |
| `power_cap_sweep.sh` | decode speed against the amdgpu `power1_cap` (50 to 30 W), restores the original cap on exit | sudo; **untested on this box** |
| `gpu_sysfs.sh` | sourced helper: finds the dGPU's sysfs and hwmon directories (read-only) | |

**Read before clearing BD PROCHOT.** It is the SMC's protective signal. On this machine the SMC asserts it near 104 to
106 C and pins the CPU at 800 MHz; clearing it removes that safety net, and 45 W runs have reached 101 to 104 C. Run it
under a junction-temperature watchdog and keep the GPU cap at 40 W or lower.
