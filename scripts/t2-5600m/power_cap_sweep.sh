#!/usr/bin/env bash
# Decode speed against the amdgpu power cap, to find tokens/s per watt on the 5600M.
# The ONLY thing this script writes is hwmon power1_cap (via sudo tee), and it restores the original value on exit.
# If power1_cap does not exist the kernel/GPU exposes no cap, which is itself the answer; the script then stops.
# It never touches pp_od_clk_voltage or the DPM level, and never sets a cap outside power1_cap_min..max.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gpu_sysfs.sh"

BENCH="${LLAMA_BENCH:-$LLAMA_BUILD/bin/llama-bench}"
MODEL="${MODEL:-$HOME/models/ornith-1.0-9b-Q4_K_M.gguf}"
CAPS="${CAPS:-50 45 40 35 30}"        # watts, tested in this order then reversed
OUT="${OUT:-$HERE/results}"; mkdir -p "$OUT"
STAMP=$(date +%Y%m%dT%H%M%S)

find_amd_gpu || { echo "no discrete AMD GPU in sysfs"; exit 1; }
CAPF="$GPU_HWMON/power1_cap"
[ -r "$CAPF" ] || { echo "power1_cap is not exposed on this GPU/kernel: no power cap available through hwmon."; exit 0; }
ORIG=$(hw power1_cap); MIN=$(hw power1_cap_min); MAX=$(hw power1_cap_max)
echo "power1_cap=$ORIG uW (min ${MIN:-?}, max ${MAX:-?})"
[ -n "$ORIG" ] || exit 1

SP=""; KA=""
restore() {
  [ -n "$SP" ] && kill $SP 2>/dev/null
  [ -n "$KA" ] && kill $KA 2>/dev/null
  if echo "$ORIG" | sudo tee "$CAPF" >/dev/null; then echo "restored power1_cap=$ORIG"; else echo "RESTORE FAILED: run: echo $ORIG | sudo tee $CAPF" >&2; fi
}
trap 'exit 130' INT TERM HUP
sudo -v || exit 1                                   # ask for the password up front
trap restore EXIT                                # only once sudo works: nothing to restore before that
( while sudo -n -v 2>/dev/null; do sleep 60; done ) & KA=$!   # keep the sudo ticket alive for a long sweep

IDX=$(amd_vulkan_index); [ -n "${IDX:-}" ] && export GGML_VK_VISIBLE_DEVICES="$IDX"
# REQUIRED memory placement: without this the model lands in GTT (system RAM) over PCIe. See README.md.
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM="${GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM:-1}"
CSV="$OUT/power_cap_$STAMP.csv"
echo "pass,cap_W,readback_W,tg256_mean_tok_s,mean_W,tok_s_per_W,max_edge_C,max_junction_C,mean_sclk_MHz,mean_mclk_MHz" > "$CSV"

ORDER=($CAPS); for ((i=${#ORDER[@]}-1; i>=0; i--)); do ORDER+=("${ORDER[$i]}"); done
pass=0
for w in "${ORDER[@]}"; do
  pass=$((pass+1)); uw=$((w*1000000))
  if { [ -n "$MIN" ] && [ "$uw" -lt "$MIN" ]; } || { [ -n "$MAX" ] && [ "$uw" -gt "$MAX" ]; }; then echo "skip ${w} W: outside cap range"; continue; fi
  echo "$uw" | sudo tee "$CAPF" >/dev/null || { echo "could not write cap"; exit 1; }
  rb=$(hw power1_cap); [ "$rb" = "$uw" ] || echo "note: wrote $uw uW, read back $rb"
  gpu_cool_to "$COOL_C"
  S="$OUT/capsamples_${STAMP}_$pass.csv"; B="$OUT/capbench_${STAMP}_$pass.csv"
  echo "$GPU_SAMPLE_HEADER" > "$S"
  ( while :; do gpu_sample >> "$S"; sleep 0.5; done ) & SP=$!
  "$BENCH" -m "$MODEL" -ngl 99 -fa 0 -p 0 -n 256 -r 3 -t 2 -o csv > "$B" 2> "$OUT/capbench_${STAMP}_$pass.err"
  kill $SP 2>/dev/null; wait $SP 2>/dev/null; SP=""
  python3 -I - "$B" "$S" "$pass" "$w" "$rb" > "$OUT/.cap_row" 2>/dev/null <<'EOF'
import csv, sys, statistics as s
rows = list(csv.DictReader(open(sys.argv[1])))
tg = [float(r["avg_ts"]) for r in rows if int(r["n_gen"]) > 0]
smp = list(csv.DictReader(open(sys.argv[2])))
def col(k):
    out = []
    for r in smp:
        try: out.append(float(r[k]))
        except Exception: pass
    return out
def bw(k):
    out = []
    for r in smp:
        try:
            if float(r["busyPct"]) >= 80: out.append(float(r[k]))
        except Exception: pass
    return out
w = bw("watts") or col("watts"); mw = s.mean(w) if w else None
t = tg[0] if tg else None
print(",".join([sys.argv[3], sys.argv[4], "%.0f" % (int(sys.argv[5]) / 1e6) if sys.argv[5] else "",
    "%.2f" % t if t else "", "%.1f" % mw if mw else "", "%.3f" % (t / mw) if t and mw else "",
    "%.1f" % max(col("edgeC")) if col("edgeC") else "", "%.1f" % max(col("junctionC")) if col("junctionC") else "",
    "%.0f" % s.mean(col("sclkMHz")) if col("sclkMHz") else "", "%.0f" % s.mean(col("mclkMHz")) if col("mclkMHz") else ""]))
EOF
  if [ -s "$OUT/.cap_row" ]; then cat "$OUT/.cap_row" | tee -a "$CSV"; else echo "cap ${w} W: run failed, see capbench_${STAMP}_$pass.err"; fi
done
echo "wrote $CSV"
