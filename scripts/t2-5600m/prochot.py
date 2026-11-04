#!/usr/bin/env python3
"""BD PROCHOT switch for this MacBookPro16,4 (i9-9880H).

MSR 0x1FC is MSR_POWER_CTL.  Bit 0 is the BD PROCHOT enable bit:
  1 = the CPU obeys PROCHOT assertions from outside the CPU (the SMC here)
  0 = it ignores them

This is the same bit ThrottleStop's "BD PROCHOT" checkbox toggles, and the same
one the t2linux wiki points at (yyearth/turnoff-BD-PROCHOT).  On this machine
it is the real clamp: the daemon's CPU *floor* of 2.5-3.5 GHz was ignored and
the CPU still fell to 800 MHz, which only a hardware assertion can do.

The write is per-CPU and volatile (a reboot restores it).  The original value
is saved so `restore` puts back exactly what was there, not a guess.

  sudo python3 prochot.py status     # read 0x1FC on every CPU, decode bit 0
  sudo python3 prochot.py off         # clear bit 0 (BD PROCHOT disabled)
  sudo python3 prochot.py on          # restore the saved value
  sudo python3 prochot.py watch 600   # log 0x1FC/0x19C/0x64F for N seconds
"""
import glob
import os
import struct
import sys
import time

MSR_POWER_CTL = 0x1FC
MSR_THERM_STATUS = 0x19C
MSR_LIMIT_REASONS = 0x64F
BD_BIT = 1 << 0
SAVE = "/tmp/prochot-1fc.saved"


def cpus():
    return sorted(glob.glob("/dev/cpu/[0-9]*/msr"),
                  key=lambda p: int(p.split("/")[3]))


def read(reg, path):
    with open(path, "rb", buffering=0) as f:
        f.seek(reg)
        return struct.unpack("<Q", f.read(8))[0]


def write(reg, val, path):
    with open(path, "wb", buffering=0) as f:
        f.seek(reg)
        f.write(struct.pack("<Q", val))


def read_all(reg):
    return [(p, read(reg, p)) for p in cpus()]


def decode(v):
    """0x1FC: bit0 BD PROCHOT enable, bit1 thermal status, bit14 lock."""
    return {
        "bd_prochot": "on " if v & BD_BIT else "off",
        "thermal_status": (v >> 1) & 1,
        "lock": (v >> 14) & 1,
    }


def status():
    vals = read_all(MSR_POWER_CTL)
    uniq = {v for _, v in vals}
    for p, v in vals[:2]:
        d = decode(v)
        print(f"  cpu {p.split('/')[3]:>2}: 0x{v:016x}  BD PROCHOT {d['bd_prochot']}"
              f"  thermal_status {d['thermal_status']}  lock {d['lock']}")
    if len(uniq) == 1:
        print(f"  all {len(vals)} cpus uniform: 0x{next(iter(uniq)):016x}  "
              f"BD PROCHOT {decode(next(iter(uniq)))['bd_prochot']}")
    else:
        print(f"  cpus differ: {sorted(hex(v) for v in uniq)}")
    return vals


def set_bit0(want_set):
    vals = read_all(MSR_POWER_CTL)
    changed = 0
    for p, v in vals:
        new = (v | BD_BIT) if want_set else (v & ~BD_BIT)
        if new != v:
            write(MSR_POWER_CTL, new, p)
            changed += 1
    time.sleep(1.0)
    back = read_all(MSR_POWER_CTL)
    held = sum(1 for _, v in back if (v & BD_BIT) == (BD_BIT if want_set else 0))
    print(f"  wrote {changed}/{len(vals)} cpus; after 1 s, {held}/{len(vals)} read back "
          f"as desired")
    if held != len(vals):
        print("  NOT HELD on every cpu - the EC/SMC re-asserts it, or the write was refused")
    return back


def restore():
    if not os.path.exists(SAVE):
        print("  no saved value; nothing to restore")
        return
    saved = {}
    for line in open(SAVE):
        path, val = line.split()
        saved[path] = int(val, 16)
    ok = 0
    for p, v in read_all(MSR_POWER_CTL):
        if p in saved and saved[p] != v:
            write(MSR_POWER_CTL, saved[p], p)
            ok += 1
    print(f"  restored the saved value on {ok} cpus")
    os.remove(SAVE)


def watch(seconds):
    end = time.time() + float(seconds)
    last = None
    print("  time            0x1FC      bd   0x19C      reasons(0x64F)  "
          "thermal_logged  power_logged")
    while time.time() < end:
        vals = read_all(MSR_POWER_CTL)
        v = vals[0][1]
        th = read(MSR_THERM_STATUS, vals[0][0])
        rs = read(MSR_LIMIT_REASONS, vals[0][0])
        cur = (v, th, rs)
        if cur != last:
            # IA32_PERF_LIMIT_REASONS (0x64F): bits 15:0 are the *current*
            # reasons, 31:16 the logged (sticky) ones.  There is no PROCHOT
            # bit in this register - PROCHOT lives in 0x19C/0x1B1.
            reasons = {
                0: "thermal", 1: "power-limit", 2: "current", 3: "reserved",
                6: "thermal1", 7: "thermal2",
                8: "PL1", 9: "PL2", 10: "PL3", 11: "PL4", 12: "PL5",
                13: "PL6", 14: "turbo-limit", 15: "turbo-at-limit",
            }
            now = [n for i, n in reasons.items() if rs >> i & 1]
            logged = [n for i, n in reasons.items() if rs >> (16 + i) & 1]
            print(f"  {time.strftime('%H:%M:%S')}  0x{v:016x}  {decode(v)['bd_prochot']}"
                  f"  0x{th:016x}  0x{rs:016x}  {','.join(now) or '-':<12}  "
                  f"{','.join(logged) or '-'}")
            last = cur
        time.sleep(0.5)


def main():
    if os.geteuid() != 0:
        raise SystemExit("needs root: sudo python3 " + sys.argv[0])
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "off":
        vals = status()
        with open(SAVE, "w") as f:
            for p, v in vals:
                f.write(f"{p} {v:016x}\n")
        print("  saved the original values to " + SAVE)
        set_bit0(False)
    elif cmd == "on":
        set_bit0(True)
        restore()
    elif cmd == "watch":
        watch(sys.argv[2] if len(sys.argv) > 2 else 60)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
