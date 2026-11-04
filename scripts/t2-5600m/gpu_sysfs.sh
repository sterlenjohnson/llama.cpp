#!/usr/bin/env bash
# Sourced by the other scripts. Locates the discrete AMD GPU's sysfs/hwmon directories and reads its sensors.
# Read-only: nothing here writes to sysfs.

# Shared settings for every script in this kit (override from the environment).
LLAMA_BUILD="${LLAMA_BUILD:-$HOME/llama.cpp/build-vulkan}"   # bin/ holds llama-bench, llama-cli, llama-perplexity, llama-server
COOL_C="${COOL_C:-65}"                                       # GPU edge temperature every run waits for

find_amd_gpu() {
  GPU_DEV=""; GPU_HWMON=""
  local c vram h
  for c in /sys/class/drm/card[0-9]*; do
    case "$c" in *-*) continue;; esac                      # skip connector entries such as card0-eDP-1
    [ -r "$c/device/vendor" ] || continue
    [ "$(cat "$c/device/vendor")" = "0x1002" ] || continue # AMD
    vram=$(cat "$c/device/mem_info_vram_total" 2>/dev/null || echo 0)
    [ "$vram" -ge 4000000000 ] || continue                 # the 5600M has 8 GB; skips integrated parts
    GPU_DEV="$c/device"
    for h in "$GPU_DEV"/hwmon/hwmon*; do [ -d "$h" ] && { GPU_HWMON="$h"; break; }; done
    [ -n "$GPU_HWMON" ] && return 0
  done
  return 1
}

# Value of a hwmon file, or empty.
hw() { cat "$GPU_HWMON/$1" 2>/dev/null; }

# Board power in watts (power1_average on most kernels, power1_input on some).
gpu_watts() {
  local uw; uw=$(hw power1_average); [ -n "$uw" ] || uw=$(hw power1_input)
  [ -n "$uw" ] && awk -v u="$uw" 'BEGIN{printf "%.1f", u/1000000}'
}
gpu_edge_c()     { local m; m=$(hw temp1_input); [ -n "$m" ] && awk -v m="$m" 'BEGIN{printf "%.1f", m/1000}'; }
gpu_junction_c() { local m; m=$(hw temp2_input); [ -n "$m" ] && awk -v m="$m" 'BEGIN{printf "%.1f", m/1000}'; }
gpu_mem_c()      { local m; m=$(hw temp3_input); [ -n "$m" ] && awk -v m="$m" 'BEGIN{printf "%.1f", m/1000}'; }
gpu_sclk_mhz()   { local h; h=$(hw freq1_input); [ -n "$h" ] && awk -v h="$h" 'BEGIN{printf "%.0f", h/1000000}'; }
gpu_mclk_mhz()   { local h; h=$(hw freq2_input); [ -n "$h" ] && awk -v h="$h" 'BEGIN{printf "%.0f", h/1000000}'; }
gpu_fan_rpm()    { hw fan1_input; }
gpu_busy_pct()   { cat "$GPU_DEV/gpu_busy_percent" 2>/dev/null; }

# Used VRAM / GTT in MiB. Call find_amd_gpu first; empty if there is no AMD device.
gpu_vram_mib()   { [ -n "${GPU_DEV:-}" ] && echo $(( $(cat "$GPU_DEV/mem_info_vram_used")/1024/1024 )); }
gpu_gtt_mib()    { [ -n "${GPU_DEV:-}" ] && echo $(( $(cat "$GPU_DEV/mem_info_gtt_used")/1024/1024 )); }

# One CSV line: epoch,watts,edgeC,junctionC,memC,sclkMHz,mclkMHz,fanRPM,busy%
gpu_sample() {
  echo "$(date +%s),$(gpu_watts),$(gpu_edge_c),$(gpu_junction_c),$(gpu_mem_c),$(gpu_sclk_mhz),$(gpu_mclk_mhz),$(gpu_fan_rpm),$(gpu_busy_pct)"
}
GPU_SAMPLE_HEADER="epoch,watts,edgeC,junctionC,memC,sclkMHz,mclkMHz,fanRPM,busyPct"

# Waits until the edge temperature is at or below $1 degrees C (default 60), for at most ~400 s.
gpu_cool_to() {
  local target="${1:-$COOL_C}" i t
  for i in $(seq 1 40); do
    t=$(gpu_edge_c); [ -n "$t" ] || return 0
    awk -v t="$t" -v g="$target" 'BEGIN{exit !(t<=g)}' && return 0
    sleep 10
  done
  echo "warning: GPU still above ${target}C after 400 s, continuing" >&2
}

# Index to put in GGML_VK_VISIBLE_DEVICES to select the discrete AMD device; empty if unknown.
#
# Do NOT hardcode this index and do NOT derive it from vulkaninfo: GGML_VK_VISIBLE_DEVICES indexes
# the order ggml gets from vkEnumeratePhysicalDevices *inside the llama.cpp process*, and that order
# is not stable across boots or driver updates. Verified 2026-10-19 (post-reboot): ggml listed
# Vulkan0 = Intel UHD 630 and Vulkan1 = AMD Radeon Pro 5600M, so =0 selects the *iGPU*; earlier
# boots recorded the reverse (=0 = the 5600M). Always ask the binary, never a saved recipe.
amd_vulkan_index() {
  local bin="${LLAMA_BIN:-$LLAMA_BUILD/bin/llama-bench}"
  [ -x "$bin" ] || return 0
  "$bin" --list-devices 2>/dev/null |
    sed -n 's/^ *Vulkan\([0-9][0-9]*\): .*\(Radeon\|RADV\|AMD\).*/\1/p' | head -1
}

# Short commit the llama.cpp *binaries* were built from.
#
# Not the same thing as the source tree's HEAD, and not interchangeable: on 2026-10-16 the tree was
# at 66ee501 while every binary in bin/ had been built from fb6acf0 (its parent). Recording
# `git rev-parse HEAD` therefore misattributes every result. The built commit is embedded in the
# version string, so ask the binary; fall back to the tree only if the binary cannot be run.
llama_build_commit() {
  local bin="${LLAMA_BIN:-$LLAMA_BUILD/bin/llama-cli}" v=""
  [ -x "$bin" ] && v=$("$bin" --version 2>&1 |
      sed -n 's/.*commit \([0-9a-f][0-9a-f]*\).*/\1/p' | head -1)
  [ -n "$v" ] || v=$(git -C "$LLAMA_BUILD" rev-parse --short HEAD 2>/dev/null)
  printf '%s' "$v"
}
