/* Fused Llama / Qwen3 attention prep (fused->attn_rope_append, #514), one
 * dispatch for what were up to seven: the per-head q / k RMSNorms (when
 * qn / kn), the q and k RoPE, the q scale and the KV append.
 * The includer defines KV_T (the cache element type) and declares nothing
 * else. Workgroup (row, head): heads [0, qh) are q's, [qh, qh + kvh) k's;
 * a k workgroup also appends the same head of v. Each step computes what
 * the decomposed ops compute, in the same order:
 *   rmsnorm_f32: lane j sums x[j]^2 (then x[j + 256]^2), subgroupAdd, the
 *     subgroups' partials in order; y = x * inv * w;
 *   rope_f32: half-split over the first nrot elements, cos/sin rows nrot
 *     wide (both halves read);
 *   scale_f32: q * qs over the whole head;
 *   kv_append(_f16): k and v rows to row qp + row of the caches.
 * Dispatch: (rows, qh + kvh). */

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Q { float q[]; };
layout(set = 0, binding = 1) buffer K { float k[]; };
layout(set = 0, binding = 2) readonly buffer V { float v[]; };
layout(set = 0, binding = 3) readonly buffer QW { float qw[]; };
layout(set = 0, binding = 4) readonly buffer KW { float kw[]; };
layout(set = 0, binding = 5) readonly buffer Cos { float cos_v[]; };
layout(set = 0, binding = 6) readonly buffer Sin { float sin_v[]; };
layout(set = 0, binding = 7) writeonly buffer KC { KV_T kc[]; };
layout(set = 0, binding = 8) writeonly buffer VC { KV_T vc[]; };

layout(push_constant) uniform Push {
    uint rows;
    uint qh;
    uint kvh;
    uint hd;
    uint nrot;
    uint qo;  /* element offsets of q, k, v, the norm weights, cos, sin */
    uint ko;
    uint vo;
    uint qwo;
    uint kwo;
    uint co;
    uint so;
    uint kco; /* cache element offsets (row 0) */
    uint vco;
    uint qp;  /* the cache row of row 0 */
    uint qn;  /* q / k normed */
    uint kn;
    uint qsc; /* q scaled by qs */
    float eps;
    float qs;
} pc;

shared float partial[256]; /* one slot per subgroup (any subgroup size) */
shared float hv[VK_ATTN_MAX_HEAD_DIM];

float xload(bool isq, uint i) {
    return isq ? q[i] : k[i];
}

void xstore(bool isq, uint i, float val) {
    if (isq) {
        q[i] = val;
    } else {
        k[i] = val;
    }
}

void main() {
    uint r = gl_WorkGroupID.x;
    uint h = gl_WorkGroupID.y;
    uint lid = gl_LocalInvocationID.x;
    uint hd = pc.hd;
    if (r >= pc.rows || h >= pc.qh + pc.kvh || hd > VK_ATTN_MAX_HEAD_DIM) {
        return;
    }
    bool isq = h < pc.qh;
    uint hh = isq ? h : h - pc.qh;
    uint base = isq ? pc.qo + (r * pc.qh + hh) * hd : pc.ko + (r * pc.kvh + hh) * hd;

    float x0 = lid < hd ? xload(isq, base + lid) : 0.0;
    float x1 = lid + 256u < hd ? xload(isq, base + lid + 256u) : 0.0;
    if (isq ? pc.qn != 0u : pc.kn != 0u) { /* uniform per workgroup */
        float sum = 0.0;
        sum = fma(x0, x0, sum);
        sum = fma(x1, x1, sum);
        sum = subgroupAdd(sum);
        if (subgroupElect()) {
            partial[gl_SubgroupID] = sum;
        }
        barrier();
        float total = 0.0;
        for (uint i = 0u; i < gl_NumSubgroups; i++) {
            total += partial[i];
        }
        float inv = inversesqrt(total / float(hd) + pc.eps);
        uint wo = isq ? pc.qwo : pc.kwo;
        if (lid < hd) {
            x0 = x0 * inv * (isq ? qw[wo + lid] : kw[wo + lid]);
        }
        if (lid + 256u < hd) {
            x1 = x1 * inv * (isq ? qw[wo + lid + 256u] : kw[wo + lid + 256u]);
        }
    }
    if (lid < hd) {
        hv[lid] = x0;
    }
    if (lid + 256u < hd) {
        hv[lid + 256u] = x1;
    }
    barrier();

    /* RoPE: pair (i, i + half) of the first nrot elements */
    uint half_rot = pc.nrot >> 1u;
    uint cb = pc.co + r * pc.nrot;
    uint sb = pc.so + r * pc.nrot;
    float y0 = x0;
    float y1 = x1;
    for (uint t = 0u; t < 2u; ++t) {
        uint i = lid + t * 256u;
        if (i >= pc.nrot) {
            continue;
        }
        float o;
        if (i < half_rot) {
            float a = hv[i];
            float b = hv[i + half_rot];
            o = a * cos_v[cb + i] - b * sin_v[sb + i];
        } else {
            uint p = i - half_rot;
            float a = hv[p];
            float b = hv[i];
            o = b * cos_v[cb + i] + a * sin_v[sb + i];
        }
        if (t == 0u) {
            y0 = o;
        } else {
            y1 = o;
        }
    }
    if (isq && pc.qsc != 0u) {
        y0 = y0 * pc.qs;
        y1 = y1 * pc.qs;
    }
    if (lid < hd) {
        xstore(isq, base + lid, y0);
    }
    if (lid + 256u < hd) {
        xstore(isq, base + lid + 256u, y1);
    }
    if (isq) {
        return;
    }
    uint cbase = (pc.qp + r) * pc.kvh * hd + hh * hd;
    uint vbase = pc.vo + (r * pc.kvh + hh) * hd;
    if (lid < hd) {
        kc[pc.kco + cbase + lid] = KV_T(y0);
        vc[pc.vco + cbase + lid] = KV_T(v[vbase + lid]);
    }
    if (lid + 256u < hd) {
        kc[pc.kco + cbase + lid + 256u] = KV_T(y1);
        vc[pc.vco + cbase + lid + 256u] = KV_T(v[vbase + lid + 256u]);
    }
}
