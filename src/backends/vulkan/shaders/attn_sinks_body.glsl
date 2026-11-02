/* Shared body of attention_sinks_{f32,f16}.comp (#243, GPT-OSS): the
 * causal (+ sliding-window) MQA/GQA attention of attention_f32.comp, one
 * workgroup per (q_pos, q_head), with the head's sink logit as one more key
 * whose value row is zero: the online softmax starts at m = sink, l = 1 and
 * nothing accumulated, so out = sum_s e^{s_s} v_s / (e^{sink} + sum_s e^{s_s}).
 * The includer declares Q, K, V, S (sinks), Out, the push block, and
 * LOAD_K(i) / LOAD_V(i) as floats. head_dim <= 512. */

layout(local_size_x = 256) in;

shared float q_sh[VK_ATTN_MAX_HEAD_DIM];
shared float red[256];
shared float w_sh[256];

void main() {
    uint q_pos  = gl_WorkGroupID.x;
    uint q_head = gl_WorkGroupID.y;
    uint lid    = gl_LocalInvocationID.x;
    if (q_pos >= pc.n_q || q_head >= pc.n_q_heads || pc.head_dim == 0 ||
        pc.head_dim > VK_ATTN_MAX_HEAD_DIM) {
        return;
    }
    uint kv_group = pc.n_q_heads / pc.n_kv_heads;
    uint kv_head  = q_head / kv_group;
    uint q_abs    = pc.q_offset_abs + q_pos;
    uint s_lo     = 0;
    if (pc.sliding_window > 0 && q_abs + 1 > pc.sliding_window) {
        s_lo = q_abs + 1 - pc.sliding_window;
    }
    uint hd       = pc.head_dim;
    uint out_base = pc.out_offset + (q_pos * pc.n_q_heads + q_head) * hd;
    /* No key: only the sink, whose value row is zero. */
    if (pc.n_kv == 0 || s_lo > q_abs || s_lo >= pc.n_kv) {
        for (uint d = lid; d < hd; d += 256) {
            out_v[out_base + d] = 0.0;
        }
        return;
    }
    uint s_hi = min(q_abs, pc.n_kv - 1);

    uint q_base = pc.q_offset + (q_pos * pc.n_q_heads + q_head) * hd;
    for (uint d = lid; d < hd; d += 256) {
        q_sh[d] = q[q_base + d];
    }
    barrier();

    float m_run  = sk[pc.sinks_offset + q_head];
    float l_run  = 1.0;
    float acc0   = 0.0; /* dim lid       */
    float acc1   = 0.0; /* dim lid + 256 */
    uint  kv_row = pc.n_kv_heads * hd;

    for (uint base = s_lo; base <= s_hi; base += 256) {
        uint  s     = base + lid;
        float score = m_run; /* below every max: contributes 0 after exp */
        bool  live  = s <= s_hi;
        if (live) {
            uint  k_base = pc.k_offset + s * kv_row + kv_head * hd;
            float sum    = 0.0;
            for (uint i = 0; i < hd; i++) {
                sum += q_sh[i] * LOAD_K(k_base + i);
            }
            score = sum;
        }
        red[lid] = score;
        barrier();
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride && red[lid + stride] > red[lid]) {
                red[lid] = red[lid + stride];
            }
            barrier();
        }
        float m_new = max(m_run, red[0]);
        barrier();

        float w  = live ? exp(score - m_new) : 0.0;
        w_sh[lid] = w;
        red[lid]  = w;
        barrier();
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride) {
                red[lid] += red[lid + stride];
            }
            barrier();
        }
        float r = exp(m_run - m_new);
        l_run   = l_run * r + red[0];
        m_run   = m_new;

        uint  chunk_n = min(255u, s_hi - base) + 1u;
        float a0      = 0.0;
        float a1      = 0.0;
        for (uint j = 0; j < chunk_n; j++) {
            uint  v_base = pc.v_offset + (base + j) * kv_row + kv_head * hd;
            float wj     = w_sh[j];
            if (lid < hd) {
                a0 += wj * LOAD_V(v_base + lid);
            }
            if (lid + 256 < hd) {
                a1 += wj * LOAD_V(v_base + lid + 256);
            }
        }
        acc0 = acc0 * r + a0;
        acc1 = acc1 * r + a1;
        barrier();
    }

    float inv_l = 1.0 / l_run;
    if (lid < hd) {
        out_v[out_base + lid] = acc0 * inv_l;
    }
    if (lid + 256 < hd) {
        out_v[out_base + lid + 256] = acc1 * inv_l;
    }
}
