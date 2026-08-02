/*
 * src/backends/vulkan/sequence.c — batched-submit command sequencing, hazards, dispatch, and
 * timing.
 *
 * Layer: BACKEND (vulkan). Split from the former monolithic backend.c;
 * pure moves, no behavior change.
 */
#include "vk_internal.h"

/* ---- Sequence core ---------------------------------------------------- */

void vk_seq_flush(struct vk_state *st) {
    if (st == nullptr || !st->seq_open) {
        return;
    }
    st->stat_flushes++;
    st->seq_open       = false;
    st->seq_dispatches = 0;
    st->n_dirty        = 0;
    bool ok            = st->fn.EndCommandBuffer(st->seq_cmd) == VK_SUCCESS;
    if (ok) {
        VkSubmitInfo    submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                  .commandBufferCount = 1,
                                  .pCommandBuffers    = &st->seq_cmd};
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = st->fn.QueueSubmit(st->queue, 1, &submit, st->seq_fence) == VK_SUCCESS &&
             st->fn.WaitForFences(st->device, 1, &st->seq_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        st->stat_wait_ns += (uint64_t) (t1.tv_sec - t0.tv_sec) * 1000000000ull +
                            (uint64_t) (t1.tv_nsec - t0.tv_nsec);
        (void) st->fn.ResetFences(st->device, 1, &st->seq_fence);
    }
    if (!ok) {
        /* Loud failure: outputs of the dropped batch are undefined and the
         * parity/token gates will catch it — same policy as the resolved
         * kernels. */
        fprintf(stderr, "geist vulkan: sequence flush failed — batch dropped\n");
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "vulkan: sequence flush failed");
    }
    if (ok && st->profile_enabled && st->ts_count > 1) {
        uint64_t ts[VK_SEQ_MAX_DISPATCH + 8];
        if (st->fn.GetQueryPoolResults(st->device,
                                       st->ts_pool,
                                       0,
                                       st->ts_count,
                                       sizeof(uint64_t) * st->ts_count,
                                       ts,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
            VK_SUCCESS) {
            for (uint32_t i = 1; i < st->ts_count; ++i) {
                const uint64_t d = ts[i] - ts[i - 1];
                st->prof_ns[st->ts_pipe[i]] += (uint64_t) ((double) d * (double) st->ts_period_ns);
                st->prof_calls[st->ts_pipe[i]]++;
            }
        }
    }
    st->ts_count = 0;
    (void) st->fn.ResetDescriptorPool(st->device, st->seq_pool, 0);
    st->xring_used = 0;
}

[[nodiscard]] enum geist_status vk_seq_open_cmd(struct vk_state *st) {
    if (st->seq_open) {
        return GEIST_OK;
    }
    st->seq_cmd_idx                = 0;
    st->seq_in_cmd                 = 0;
    st->seq_cmd                    = st->seq_cmds[0];
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (st->fn.BeginCommandBuffer(st->seq_cmd, &begin) != VK_SUCCESS) {
        geist_backend_set_error(st->backend, GEIST_E_BACKEND, "vulkan: seq begin failed");
        return GEIST_E_BACKEND;
    }
    st->seq_open = true;
    if (st->profile_enabled) {
        st->fn.CmdResetQueryPool(st->seq_cmd, st->ts_pool, 0, VK_SEQ_MAX_DISPATCH + 8);
        st->fn.CmdWriteTimestamp(st->seq_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, st->ts_pool, 0);
        st->ts_count = 1;
    }
    return GEIST_OK;
}

/* Rolling submission: close the open command buffer, hand it to the GPU
 * WITHOUT waiting, and keep recording in the next ring slot. Keeps the
 * GPU fed (and boosted) while the CPU encodes the rest of the token. */
static void vk_seq_roll(struct vk_state *st) {
    if (!st->seq_open || st->seq_in_cmd == 0 || st->seq_cmd_idx + 1 >= VK_SEQ_CMDBUFS) {
        return;
    }
    if (st->fn.EndCommandBuffer(st->seq_cmd) != VK_SUCCESS) {
        return; /* keep recording — flush will fail loudly if truly broken */
    }
    VkSubmitInfo submit = {.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers    = &st->seq_cmd};
    (void) st->fn.QueueSubmit(st->queue, 1, &submit, VK_NULL_HANDLE);
    st->seq_cmd_idx++;
    st->seq_cmd                    = st->seq_cmds[st->seq_cmd_idx];
    st->seq_in_cmd                 = 0;
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    (void) st->fn.BeginCommandBuffer(st->seq_cmd, &begin);
}

/* Record a per-dispatch timestamp attributed to `slot` (pipe index, or
 * VK_PIPE_COUNT for transfer copies). */
void vk_prof_stamp(struct vk_state *st, uint32_t slot) {
    if (!st->profile_enabled || st->ts_count >= VK_SEQ_MAX_DISPATCH + 8) {
        return;
    }
    st->fn.CmdWriteTimestamp(
            st->seq_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, st->ts_pool, st->ts_count);
    st->ts_pipe[st->ts_count] = (uint8_t) slot;
    st->ts_count++;
}

/* Execution barrier between dependent dispatches/copies in the batch. One
 * global memory barrier — coarse but correct on a single compute queue.
 * ponytail: per-buffer barriers if the profiler ever blames this. */
static void vk_seq_barrier(struct vk_state *st) {
    st->n_dirty = 0;
    st->stat_barriers++;
    static _Atomic int no_bar = -1;
    if (no_bar < 0) {
        no_bar = getenv("GEIST_VK_NO_BARRIER") != nullptr; /* perf probe: WRONG results */
    }
    if (no_bar > 0) {
        return;
    }
    const VkMemoryBarrier mb = {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    st->fn.CmdPipelineBarrier(st->seq_cmd,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0,
                              1,
                              &mb,
                              0,
                              nullptr,
                              0,
                              nullptr);
}

/* Barrier iff the new accesses conflict with anything recorded since the
 * last barrier, then record them. acc == nullptr → conservative (always
 * barrier when the batch is non-empty, like the pre-tracking behavior). */
void vk_seq_hazard(struct vk_state              *st,
                   const VkDescriptorBufferInfo *infos,
                   const struct vk_access       *acc,
                   uint32_t                      n) {
    if (st->seq_dispatches == 0) {
        st->n_dirty = 0;
    }
    bool conflict = acc == nullptr && st->seq_dispatches > 0;
    if (acc != nullptr) {
        for (uint32_t i = 0; i < n && !conflict; ++i) {
            for (uint32_t d = 0; d < st->n_dirty; ++d) {
                const struct vk_dirty *e = &st->dirty[d];
                if (e->buf != infos[i].buffer || (!e->write && !acc[i].write)) {
                    continue; /* different buffer or read-read */
                }
                if (acc[i].lo < e->hi && e->lo < acc[i].hi) {
                    conflict = true;
                    break;
                }
            }
        }
        if (!conflict && st->n_dirty + n > VK_DIRTY_CAP) {
            conflict = true; /* table full — degrade to the old behavior */
        }
    }
    if (conflict && st->seq_dispatches > 0) {
        vk_seq_barrier(st);
    } else if (st->seq_dispatches > 0) {
        st->stat_barriers_elided++;
    }
    if (acc != nullptr) {
        for (uint32_t i = 0; i < n && st->n_dirty < VK_DIRTY_CAP; ++i) {
            st->dirty[st->n_dirty++] = (struct vk_dirty) {.buf   = infos[i].buffer,
                                                          .lo    = acc[i].lo,
                                                          .hi    = acc[i].hi,
                                                          .write = acc[i].write};
        }
    } else {
        st->n_dirty = 0; /* unknown writes — next dispatch must barrier */
    }
}

/* Append one compute dispatch to the open sequence. infos[] length must
 * equal the pipeline's binding count. */
[[nodiscard]] enum geist_status vk_seq_dispatch_acc(struct geist_backend         *be,
                                                    enum vk_pipe                  pipe,
                                                    const VkDescriptorBufferInfo *infos,
                                                    const struct vk_access       *acc,
                                                    const void                   *push,
                                                    uint32_t                      push_bytes,
                                                    uint32_t                      gx,
                                                    uint32_t                      gy,
                                                    uint32_t                      gz) {
    struct vk_state *st = be->state;
    if (st->seq_dispatches >= VK_SEQ_MAX_DISPATCH) {
        vk_seq_flush(st);
    }
    enum geist_status s = vk_seq_open_cmd(st);
    if (s != GEIST_OK) {
        return s;
    }
    const uint32_t nbind = vk_pipe_nbind[pipe];
    /* descriptor-set cache: same (nbind, buffers) tuple recurs every token */
    uint64_t h = 1469598103934665603ull ^ nbind;
    for (uint32_t i = 0; i < nbind; ++i) {
        h = (h ^ (uint64_t) infos[i].buffer) * 1099511628211ull;
    }
    if (h == 0) {
        h = 1;
    }
    VkDescriptorSet set    = VK_NULL_HANDLE;
    uint32_t        islot  = UINT32_MAX; /* first reusable slot (empty/tombstone) */
    bool            iempty = false;
    uint32_t        slot   = (uint32_t) (h & (VK_DSET_CACHE - 1));
    for (uint32_t probe = 0; probe < 16; ++probe, slot = (slot + 1) & (VK_DSET_CACHE - 1)) {
        const uint64_t k = st->dset_cache[slot].key;
        if (k == h) {
            set = st->dset_cache[slot].set;
            st->stat_dset_hits++;
            break;
        }
        if (islot == UINT32_MAX && (k == 0 || k == UINT64_MAX)) {
            islot  = slot;
            iempty = k == 0;
        }
        if (k == 0) {
            break;
        }
    }
    if (set == VK_NULL_HANDLE) {
        st->stat_dset_miss++;
        slot = islot != UINT32_MAX ? islot : slot;
        /* tombstoned slots reuse their old set object; empty slots get a
         * fresh one from the cache pool */
        const bool cacheable = islot != UINT32_MAX;
        if (cacheable && !iempty && st->dset_cache[slot].set != VK_NULL_HANDLE) {
            set                      = st->dset_cache[slot].set;
            st->dset_cache[slot].key = h;
        }
        if (set == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo ainfo2 = {
                    .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool     = cacheable ? st->dset_cache_pool : st->seq_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts        = &st->seq_dlayouts[nbind - 2]};
            if (st->fn.AllocateDescriptorSets(st->device, &ainfo2, &set) != VK_SUCCESS) {
                /* cache pool dry — transient set from the per-flush pool */
                ainfo2.descriptorPool = st->seq_pool;
                if (st->fn.AllocateDescriptorSets(st->device, &ainfo2, &set) != VK_SUCCESS) {
                    vk_seq_flush(st);
                    s = vk_seq_open_cmd(st);
                    if (s != GEIST_OK ||
                        st->fn.AllocateDescriptorSets(st->device, &ainfo2, &set) != VK_SUCCESS) {
                        geist_backend_set_error(
                                be, GEIST_E_BACKEND, "vulkan: descriptor alloc failed");
                        return GEIST_E_BACKEND;
                    }
                }
            } else if (cacheable) {
                st->dset_cache[slot] = (struct vk_dset_entry) {.key = h, .set = set};
            }
        }
        VkWriteDescriptorSet writes[6];
        for (uint32_t i = 0; i < nbind; ++i) {
            writes[i] = (VkWriteDescriptorSet) {.sType  = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                .dstSet = set,
                                                .dstBinding      = i,
                                                .descriptorCount = 1,
                                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                .pBufferInfo    = &infos[i]};
        }
        st->fn.UpdateDescriptorSets(st->device, nbind, writes, 0, nullptr);
    }
    vk_seq_hazard(st, infos, acc, nbind);
    st->fn.CmdBindPipeline(st->seq_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, st->pipes[pipe]);
    st->fn.CmdBindDescriptorSets(st->seq_cmd,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 st->seq_playouts[nbind - 2],
                                 0,
                                 1,
                                 &set,
                                 0,
                                 nullptr);
    st->fn.CmdPushConstants(st->seq_cmd,
                            st->seq_playouts[nbind - 2],
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0,
                            push_bytes,
                            push);
    st->fn.CmdDispatch(st->seq_cmd, gx, gy, gz);
    st->seq_dispatches++;
    st->seq_in_cmd++;
    st->stat_dispatches++;
    vk_prof_stamp(st, (uint32_t) pipe);
    if (st->seq_in_cmd >= VK_SEQ_ROTATE) {
        vk_seq_roll(st);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status vk_seq_dispatch(struct geist_backend         *be,
                                                enum vk_pipe                  pipe,
                                                const VkDescriptorBufferInfo *infos,
                                                const void                   *push,
                                                uint32_t                      push_bytes,
                                                uint32_t                      gx,
                                                uint32_t                      gy,
                                                uint32_t                      gz) {
    return vk_seq_dispatch_acc(be, pipe, infos, nullptr, push, push_bytes, gx, gy, gz);
}
