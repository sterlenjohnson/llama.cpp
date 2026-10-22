#pragma once
// RAII scope for the host-overhead profiler (GGML_HOST_PROF=1), see ggml_hp_* in ggml.h
#include "ggml.h"
#ifdef __cplusplus
struct ggml_hp_scope {
    int     slot;
    int64_t t0;
    explicit ggml_hp_scope(int s) : slot(s), t0(ggml_hp_now()) {}
    ~ggml_hp_scope() { ggml_hp_add(slot, t0); }
    ggml_hp_scope(const ggml_hp_scope &) = delete;
    ggml_hp_scope & operator=(const ggml_hp_scope &) = delete;
};
enum {
    GGML_HP_DECODE = 0, GGML_HP_PROCESS_UBATCH, GGML_HP_BUILD, GGML_HP_SET_INPUTS, GGML_HP_GRAPH_COMPUTE,
    GGML_HP_SCHED_COPY_INPUT, GGML_HP_MOE_CB, GGML_HP_MOE_PLAN, GGML_HP_VK_GRAPH_COMPUTE, GGML_HP_VK_SYNC,
    GGML_HP_VK_FENCE, GGML_HP_LOGITS, GGML_HP_CPU_COMPUTE, GGML_HP_SAMPLE, GGML_HP_SCHED_PREV_SYNC,
    GGML_HP_MOE_IDS, GGML_HP_VK_BUILD, GGML_HP_VK_GET, GGML_HP_VK_SET, GGML_HP_LLAMA_SYNC, GGML_HP_SPLITS,
    GGML_HP_VK_SUBMIT, GGML_HP_POOL_WAKE, GGML_HP_SUBMIT_AFTER,
    GGML_HP_MOE_CB_SYNC, GGML_HP_MOE_CB_IDS, GGML_HP_MOE_CB_ISSUE,
};
#define GGML_HP(slot) ggml_hp_scope ggml_hp_scope_##slot(slot)
#endif
