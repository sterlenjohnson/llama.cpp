#include "ggml-hp.h"
#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <cstring>
#include <cstdint>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::unordered_set<int32_t> hot_keys;
    std::vector<int32_t> layer_quota; // per-layer slot cap; 0 = unlimited
    void set_hot(const std::vector<std::pair<int32_t,int32_t>>& ranked, int32_t n_expert_) {
        hot_keys.clear();
        for (const auto & pr : ranked) {
            hot_keys.insert(pr.first * n_expert_ + pr.second);
        }
    }
    // keep only the top `per_layer` pairs of each layer hot: the hot set has to
    // be SMALLER than the layer's quota, or the layer has no room left for the
    // experts the routing touches cold, and a cold request then has to evict a
    // hot one - the opposite of what the profile is for.
    void set_hot_top(const std::vector<std::pair<int32_t,int32_t>>& ranked, int32_t n_expert_, int32_t per_layer, int32_t n_layer_) {
        hot_keys.clear();
        std::vector<int32_t> taken((size_t) n_layer_, 0);
        for (const auto & pr : ranked) {
            if (pr.first < 0 || pr.first >= n_layer_) continue;
            if (taken[pr.first] >= per_layer) continue;
            taken[pr.first]++;
            hot_keys.insert(pr.first * n_expert_ + pr.second);
        }
    }
    void set_quotas(const std::vector<int32_t>& q) {
        layer_quota = q;
    }
    int32_t count_slots_layer(int32_t il) const {
        if (il < 0 || il >= (int32_t) slot_map.size() || !slot_map[il]) return 0;
        int32_t c = 0;
        for (int32_t e = 0; e < n_expert; ++e) {
            if (slot_map[il][e] >= 0) ++c;
        }
        return c;
    }

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    // a slot may be evicted if it is empty, holds an expert of another layer, or
    // holds an expert that this request did not select - including the earlier
    // fills of the same request
    bool evictable(int32_t s, int32_t il) const {
        const int32_t key = key_of[s];
        if (key < 0) {
            return true;
        }
        return key / n_expert != il || seen[key % n_expert] != seen_gen;
    }

    // The SLRU's own order: the probation segment first, then the protected one,
    // and inside a segment the least recently used first.
    //
    // Two refinements ride on that order and never change it:
    //  - the expert profile's hot set is evicted only when there is nothing else
    //    to evict, so a hot expert survives a cold request while it can;
    //  - the per-layer quota makes a layer that is at its share of the slots
    //    evict one of its own cold slots first, so one layer cannot eat another's.
    //
    // Neither refinement may refuse an eviction the plain SLRU would have made:
    // they only change WHICH slot is chosen, and the last pass is upstream's rule
    // unchanged.  With no profile and no quota this is upstream's victim() exactly.
    int32_t victim(int32_t il) const {
        const bool at_quota = (il >= 0 && il < (int32_t) layer_quota.size() && layer_quota[il] > 0 &&
                               count_slots_layer(il) >= layer_quota[il]);
        // 1. the layer's own cold slot, when the layer is at its quota
        if (at_quota) {
            for (int segment = 0; segment < 2; ++segment) {
                for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                    const int32_t key = key_of[s];
                    if (key < 0 || key / n_expert != il || !evictable(s, il)) {
                        continue;
                    }
                    if (!hot_keys.count(key)) {
                        return s;
                    }
                }
            }
        }
        // 2. the coldest slot that is not hot
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                if (!evictable(s, il)) {
                    continue;
                }
                if (key < 0 || !hot_keys.count(key)) {
                    return s;
                }
            }
        }
        // 3. upstream's rule: the coldest slot this request did not select
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                if (evictable(s, il)) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    int32_t n_layers = 0;
    int32_t n_expert = 0;
    std::vector<std::pair<int32_t,int32_t>> profile_ranked;
    const char * profile_path = nullptr;
    const char * profile_save_path = nullptr;
    bool profile_loaded = false;
    std::vector<uint64_t> expert_counts;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // LLAMA_MOE_CACHE_SYNC_LEGACY=1: slot maps and ids in pageable memory and a scheduler-style synchronization before the
    // readback, this is the old path, kept for A/B comparisons
    // otherwise the slot maps and the ids are in the pinned host buffer of the GPU, so the readback of the ids and the copy
    // of the slot map are plain async copies, the only blocking wait of a cached layer is the readback of its ids
    bool sync_legacy = false;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // pinned destination of the ids readback, nullptr in the legacy path
    ggml_backend_buffer_ptr buf_ids;
    int32_t * ids_pinned = nullptr;
    size_t    ids_pinned_n = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const char * profile_path, const char * profile_save_path) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()), profile_path(profile_path), profile_save_path(profile_save_path), profile_loaded(false) {
        (void) profile_path;
        {
            const char * env = getenv("LLAMA_MOE_CACHE_SYNC_LEGACY");
            sync_legacy = env != nullptr && atoi(env) != 0;
        }
        n_layers = (int32_t) model.layers.size();
        n_expert = (int32_t) model.hparams.n_expert;
        expert_counts.assign((size_t)n_layers * n_expert, 0);
        LLAMA_LOG_INFO("%s: size = %.1f MiB, n_expert = %d, n_expert_used = %d, n_layer = %zu\n",
                       __func__, size/1024.0/1024.0, model.hparams.n_expert, n_expert_used, layers.size());
        // bounded step: profile loading is deferred (see ensure_profile_loaded), to avoid dangling pointers during fit.
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            g.slru.set_hot(profile_ranked, n_expert);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        // the pinned host buffer of the first GPU, so that the backend can copy the slot maps without staging them
        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        if (!sync_legacy) {
            ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(devices[0].backend));
            if (buft_pinned != nullptr) {
                buft_host = buft_pinned;
            }
        }
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }

            // the ids of a ubatch that uses the cache: at most max_batch tokens, see get_slot_map
            if (!sync_legacy && buft_host != ggml_backend_cpu_buffer_type()) {
                ids_pinned_n = (size_t) max_batch*n_expert_used;
                buf_ids.reset(ggml_backend_buft_alloc_buffer(buft_host, ids_pinned_n*sizeof(int32_t)));
                if (buf_ids) {
                    ids_pinned = (int32_t *) ggml_backend_buffer_get_base(buf_ids.get());
                } else {
                    ids_pinned_n = 0;
                }
            }
        }

        {
            size_t slots = 0;
            size_t slot_bytes = 0;
            size_t cached_layers = 0;
            for (const group & g : groups) {
                if (g.n_slots == 0) {
                    continue;
                }
                slots += (size_t) g.n_slots;
                cached_layers += g.layers.size();
                for (const ggml_tensor * t : g.ref) {
                    slot_bytes += (size_t) t->nb[2] * g.n_slots;
                }
            }
            LLAMA_LOG_INFO("llama_moe_cache: budget = %.1f MiB, %zu slots (%.1f MiB of experts) over %zu layers, "
                           "host experts = %.1f MiB\n",
                           size/1024.0/1024.0, slots, slot_bytes/1024.0/1024.0, cached_layers,
                           [&] {
                               size_t h = 0;
                               for (const device & d : devices) {
                                   h += d.host_bytes;
                               }
                               return h/1024.0/1024.0;
                           }());
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, slot maps in %s%s\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0,
                ggml_backend_buft_name(buft_host), sync_legacy ? " (LLAMA_MOE_CACHE_SYNC_LEGACY)" : "");
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                    // the layers the cache covers, so an expert profile can be pruned to them
                    std::string ls;
                    for (size_t i = 0; i < g.layers.size(); ++i) {
                        ls += (i ? "," : "") + std::to_string(g.layers[i]);
                    }
                    LLAMA_LOG_INFO("%s:   cached layers: %s\n", __func__, ls.c_str());
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
        if (profile_save_path) {
            // write STRP profile: rank (layer,expert) by count descending
            std::vector<std::pair<uint64_t,int32_t>> ranked; // count, index
            const size_t total = (size_t)n_layers * n_expert;
            for (size_t idx = 0; idx < total; ++idx) {
                if (expert_counts[idx] > 0) ranked.push_back({expert_counts[idx], (int32_t)idx});
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b){ return a.first > b.first; });
            std::FILE * f = std::fopen(profile_save_path, "wb");
            if (f) {
                const char magic[4] = {'S','T','R','P'};
                std::fwrite(magic,1,4,f);
                uint32_t hdr[5] = {1, (uint32_t)n_layers, (uint32_t)n_expert, (uint32_t)ranked.size(), (uint32_t)ranked.size()};
                std::fwrite(hdr,4,5,f);
                std::vector<uint16_t> raw;
                raw.reserve(ranked.size()*2);
                for (const auto & r : ranked) {
                    int32_t l = r.second / n_expert;
                    int32_t e = r.second % n_expert;
                    raw.push_back((uint16_t)l);
                    raw.push_back((uint16_t)e);
                }
                if (!raw.empty()) std::fwrite(raw.data(),2,raw.size(),f);
                std::fclose(f);
                fprintf(stderr, "moe_cache: saved profile %s: %zu ranked pairs\n", profile_save_path, ranked.size());
            } else {
                fprintf(stderr, "moe_cache: could not save profile %s\n", profile_save_path);
            }
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    void ensure_profile_loaded() {
        if (profile_loaded || !profile_path) return;
        profile_loaded = true;
        std::string profile_str = profile_path;
        std::FILE * f = std::fopen(profile_str.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "moe_cache: could not open profile %s\n", profile_str.c_str());
            return;
        }
        char magic[4]={0,0,0,0};
        uint32_t hdr[5]={0,0,0,0,0};
        if (std::fread(magic,1,4,f)!=4 || std::fread(hdr,4,5,f)!=5 || memcmp(magic,"STRP",4)!=0) {
            std::fclose(f); return;
        }
        const uint32_t nl=hdr[1], ne=hdr[2], n_ranked=hdr[4];
        // The profile may cover FEWER layers than the model has.  The engine
        // counts the MTP head as a layer: the 35B has 40 trunk layers plus
        // blk.40.nextn (qwen35moe.nextn_predict_layers = 1), so model.layers
        // has 41 entries and n_layer_all prints 41, while a profile built from
        // the trunk routing has 40.  A profile that names a layer the model
        // does not have is an error; one that omits the MTP layer is not -
        // llama.cpp ignores the MTP head's tensors anyway.
        if ((int64_t)nl > n_layers || (int64_t)ne != n_expert) {
            fprintf(stderr, "moe_cache: profile is %u layers x %u experts, the model has %d layers x %d experts - not used\n",
                    nl, ne, n_layers, n_expert);
            std::fclose(f); return;
        }
        profile_ranked.assign(n_ranked, {0,0});
        std::vector<uint16_t> raw((size_t)n_ranked*2);
        if (n_ranked==0 || std::fread(raw.data(),2,(size_t)n_ranked*2,f)==(size_t)n_ranked*2) {
            for (uint32_t i=0;i<n_ranked;i++) {
                profile_ranked[i] = {(int32_t)raw[(size_t)i*2], (int32_t)raw[(size_t)i*2+1]};
            }
            fprintf(stderr, "moe_cache: loaded profile %s: %u ranked pairs\n", profile_str.c_str(), n_ranked);
        } else {
            profile_ranked.clear();
        }
        std::fclose(f);
        // count profile pairs per layer
        std::vector<int32_t> pair_count(n_layers, 0);
        for (const auto & pr : profile_ranked) {
            if (pr.first >= 0 && pr.first < n_layers) pair_count[pr.first]++;
        }
        for (auto & g : groups) {
            // size per-layer quotas for this group's layers
            std::vector<int32_t> quotas(n_layers, 0);
            int64_t total_pairs = 0;
            for (int32_t il : g.layers) {
                if (il >= 0 && il < n_layers) total_pairs += pair_count[il];
            }
            if (total_pairs > 0) {
                // The quota is each layer's SHARE of the slots, scaled by how many
                // of the profile's pairs belong to that layer.  It must stay a
                // share and not become the hot count: a layer needs slots for the
                // experts the routing touches cold as well as for the ones it
                // touches hot, and a quota equal to the hot count leaves no room
                // for the cold ones - every cold request then has to evict a hot
                // expert, which is the opposite of what the profile is for.
                //
                // The share is taken over the profile's pairs for the group's
                // layers, but the quota is floored at the number of hot pairs
                // that layer can actually hold, and the profile's pairs beyond
                // the quota stay hot-but-evictable.
                int32_t remaining = g.n_slots;
                for (size_t idx = 0; idx < g.layers.size(); ++idx) {
                    int32_t il = g.layers[idx];
                    if (il < 0 || il >= n_layers) continue;
                    int32_t q;
                    if (idx + 1 == g.layers.size()) {
                        q = remaining;
                    } else {
                        q = (int32_t) ((int64_t) g.n_slots * pair_count[il] / total_pairs);
                        if (q < 1) q = 1;
                    }
                    if (q > remaining) q = remaining;
                    remaining -= q;
                    quotas[il] = q;
                }
                // give any leftover to the last layer
                if (!g.layers.empty()) {
                    int32_t last = g.layers.back();
                    if (last >= 0 && last < n_layers) quotas[last] += remaining;
                }
            } else {
                // no profile info: uniform split
                int32_t base = g.n_slots / (int32_t) g.layers.size();
                int32_t rem = g.n_slots - base * (int32_t) g.layers.size();
                for (int32_t il : g.layers) {
                    if (il >= 0 && il < n_layers) {
                        quotas[il] = base + (rem > 0 ? 1 : 0);
                        if (rem > 0) --rem;
                    }
                }
            }
            g.slru.set_quotas(quotas);

            // Now the hot set.  It is the top of the ranking, but capped per
            // layer so that it stays SMALLER than the layer's quota: the quota
            // is the layer's share of the slots and the hot set is the part of
            // that share worth protecting.  The rest of the share is the churn
            // pool the cold experts cycle through.
            //
            // With the profile pruned to the cache's slot count the quota and
            // the hot count came out EQUAL (the quota is the profile's share of
            // the slots, and the profile had exactly one slot's worth of pairs),
            // which left no churn pool at all: every cold request had to evict
            // a hot expert and the hit rate fell BELOW the plain LRU's.
            int32_t hot_per_layer = 0;
            for (int32_t il : g.layers) {
                if (il >= 0 && il < n_layers) hot_per_layer = std::max(hot_per_layer, quotas[il]);
            }
            // keep a FRACTION of each layer's share hot, at least one.
            // The fraction is tunable because the trade is real: more hot means
            // better protection of the experts the ranking likes, and less room
            // for the ones the routing touches cold.
            {
                const char * e = getenv("GGML_MOE_HOT_FRACTION");
                int32_t num = 1, den = 2;
                if (e) {
                    const char * sl = strchr(e, '/');
                    if (sl) { den = atoi(sl+1); num = atoi(e); }
                    else { num = atoi(e); den = 1; }
                    if (den < 1) den = 1;
                    if (num < 0) num = 0;
                }
                hot_per_layer = std::max(1, (hot_per_layer * num) / den);
            }
            g.slru.set_hot_top(profile_ranked, n_expert, hot_per_layer, n_layers);
        }
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        ensure_profile_loaded();
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        // the scheduler does not synchronize the backend before it calls back (see ggml_backend_sched_copy_input),
        // the legacy path does it here, as the scheduler did
        if (sync_legacy) {
            ggml_backend_synchronize(backend);
        }

        // this synchronization waits for the split that selects the experts and for all the earlier work of the backend:
        // the uploads below overwrite slots that earlier splits may read, and the slot maps, which the backend reads
        // from pinned memory when it executes their copies, are changed by plan
        const size_t n_ids = ggml_nelements(sel);
        int32_t * ids_data;
        if (ids_pinned != nullptr && n_ids <= ids_pinned_n) {
            ids_data = ids_pinned;
        } else {
            ids.resize(n_ids);
            ids_data = ids.data();
        }
        const int pcie_prev = ggml_backend_pcie_set_tag(GGML_PCIE_IDS);
        {
            GGML_HP(GGML_HP_MOE_IDS);
            ggml_backend_tensor_get_async(backend, sel, ids_data, 0, ggml_nbytes(sel));
            ggml_backend_synchronize(backend);
        }

        size_t n_hit = 0;
        {
            GGML_HP(GGML_HP_MOE_PLAN);
            if (!g.slru.plan(il, ids_data, n_ids, fills, n_hit)) {
                GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
            }
        }
        // profile capture: count distinct selected experts for this layer
        if (il >= 0 && il < n_layers && expert_counts.size() == (size_t)n_layers * n_expert) {
            // distinct
            std::vector<char> seen_e(n_expert, 0);
            for (size_t i = 0; i < n_ids; ++i) {
                int32_t e = ids_data[i];
                if (e >= 0 && e < n_expert && !seen_e[e]) {
                    seen_e[e] = 1;
                    expert_counts[(size_t)il * n_expert + e]++;
                }
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                ggml_backend_pcie_set_tag(GGML_PCIE_EXPERT_CACHE);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        // the backend reads a pinned slot map when the copy executes (Vulkan stages and synchronizes a pageable one here)
        ggml_backend_pcie_set_tag(GGML_PCIE_SLOT_MAP);
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        ggml_backend_pcie_set_tag(pcie_prev);

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const char * profile_path, const char * profile_save_path) :
    pimpl(new impl(model, backends, bufts, size, profile_path, profile_save_path)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
