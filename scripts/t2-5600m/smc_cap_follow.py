#!/usr/bin/env python3
"""Follow the SMC's own dGPU power request, through LACT, with no root.

The Strata work found the mechanism this box uses to shed heat: the SMC
publishes a dGPU power *step* in t2smc's smc_plimit_raw, and t2-zone-power
translates that step into a dGPU power cap.  The daemon can run in that
follow-the-SMC mode, or in a fixed mode that ignores it.

On this box the daemon runs FIXED (--dgpu-cap-w 40), so the SMC's request is
never honoured: during the 128k llama.cpp run the SMC held step 15 (its
maximum request) for 565 s and the cap never moved.  Four hundred and sixty
seconds later the SMC started cutting the CPU instead - and llama.cpp runs 26
of its expert layers on the CPU, so that cut lands directly on the decode.

This script closes that loop from the other side.  It reads the SMC's step,
maps it to a cap the way the daemon's DGpuCap.target() does, and writes that
cap through LACT's socket - no root, no systemd change, reversible by
stopping the script.

    python3 engine/linux/smc_cap_follow.py --watch        # follow, print steps
    python3 engine/linux/smc_cap_follow.py --watch --floor 30
    python3 engine/linux/smc_cap_follow.py --status       # show the mapping

The mapping mirrors t2-zone-power's DGpuCap.target(): step 0 is the card's
tdp, step 15 is the floor, 16 steps in between.  --floor overrides the
floor (the daemon's default floor is 40% of tdp = 20 W).
"""
import argparse
import json
import os
import subprocess
import sys
import time

T2_HWMON = "/sys/class/hwmon/hwmon1"
GPU_HWMON = None
LACT_SOCK = "/run/lactd.sock"
GPU_ID = os.environ.get("GPU_ID", "1002:7360-106B:021E-0000:03:00.0")
DGPU_STEPS = 16


def find_gpu():
    global GPU_HWMON
    for card in sorted(os.listdir("/sys/class/drm")):
        if not card.startswith("card"):
            continue
        base = f"/sys/class/drm/{card}/device/hwmon"
        if os.path.isdir(base):
            for h in sorted(os.listdir(base)):
                p = os.path.join(base, h)
                if os.path.exists(os.path.join(p, "power1_average")):
                    GPU_HWMON = p
                    return p
    raise SystemExit("no amdgpu hwmon found")


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def tdp():
    return int(read(os.path.join(GPU_HWMON, "power1_cap_default"), "50000000")) // 1_000_000


def cap_min():
    return int(read(os.path.join(GPU_HWMON, "power1_cap_min"), "20000000")) // 1_000_000


def cap_max():
    return int(read(os.path.join(GPU_HWMON, "power1_cap_max"), "50000000")) // 1_000_000


def smc_steps():
    """The SMC's four step bytes and its dGPU step, from smc_plimit_raw."""
    raw = read(os.path.join(T2_HWMON, "smc_plimit_raw"))
    if not raw:
        return None
    word_s, gpu_s = raw.split()
    word = int(word_s, 16)
    gpu = int(gpu_s, 16)
    b = word.to_bytes(4, "little")
    return dict(cpu=b[0] & 0x7F, igpu=b[1] & 0x7F, mem=b[2] & 0x7F,
                idle=b[3] & 0x7F, dgpu=gpu & 0x7F)


def lact(command, args):
    payload = json.dumps({"command": command, "args": args})
    try:
        out = subprocess.run(["ncat", "-U", LACT_SOCK], input=payload + "\n",
                             capture_output=True, text=True, timeout=5)
        return json.loads(out.stdout or "{}")
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return {}


def read_cap():
    return int(read(os.path.join(GPU_HWMON, "power1_cap"), "0")) // 1_000_000


def set_cap(w):
    """Write the cap through LACT and CONFIRM it - an unconfirmed change
    reverts to LACT's last confirmed value after five seconds."""
    lact("set_power_cap", {"id": GPU_ID, "cap": int(w)})
    time.sleep(0.6)
    lact("confirm_pending_config", {"command": "confirm"})
    time.sleep(0.6)
    return read_cap()


def target_watts(step, floor_w, ceil_w=None):
    """The daemon's mapping, bounded by a ceiling.

    The daemon maps step 0 to the card's tdp (50 W here) and step 15 to the
    floor.  A ceiling is needed for this box: the ladder's cap ladder showed
    45 W running the 64k prefill to 102 C, so "step 0" must not mean 50 W.
    The scale is compressed into [floor, ceil] instead.
    """
    t = tdp()
    lo = max(cap_min(), floor_w)
    hi = cap_max() if ceil_w is None else min(cap_max(), ceil_w)
    if hi < lo:
        hi = lo
    index = min(step, DGPU_STEPS - 1)
    span = hi - lo
    return lo + int(span * (DGPU_STEPS - 1 - index) / (DGPU_STEPS - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--floor", type=int, default=0,
                    help="the cap's floor in W (0 = the daemon's default, 40%% of tdp)")
    ap.add_argument("--ceil", type=int, default=0,
                    help="the cap's ceiling in W (0 = the card's tdp)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--hold", type=int, default=0,
                    help="stop following after N seconds and restore the cap")
    args = ap.parse_args()

    find_gpu()
    t, lo, hi = tdp(), cap_min(), cap_max()
    floor = args.floor if args.floor else -(-t * 4 // 10)   # ceil, as the daemon does
    ceil_w = args.ceil if args.ceil else hi

    if args.status:
        s = smc_steps() or {}
        print(f"tdp {t} W   cap range {lo}-{hi} W   floor {floor} W   ceiling {ceil_w} W")
        print(f"SMC steps now: {s}")
        print(f"current cap: {read_cap()} W")
        print("\nstep -> cap")
        for st in range(DGPU_STEPS):
            print(f"  {st:2d} -> {target_watts(st, floor, ceil_w):2d} W")
        return

    if not args.watch:
        ap.error("give --watch or --status")

    print(f"following the SMC's dGPU step, {floor}-{ceil_w} W, "
          f"writing through LACT (no root)")
    applied = None
    t0 = time.time()
    try:
        while True:
            s = smc_steps()
            if s is None:
                time.sleep(args.interval)
                continue
            want = target_watts(s["dgpu"], floor, ceil_w)
            have = read_cap()
            if want != have:
                got = set_cap(want)
                print(f"[{time.strftime('%H:%M:%S')}] SMC dgpu step {s['dgpu']:2d} "
                      f"(cpu {s['cpu']:2d}) -> cap {want} W (read back {got} W)",
                      flush=True)
                applied = want
            if args.hold and time.time() - t0 > args.hold:
                print(f"holding for {args.hold}s - restoring the cap to {hi} W")
                set_cap(hi)
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nrestoring the cap to {hi} W")
        set_cap(hi)


if __name__ == "__main__":
    main()
