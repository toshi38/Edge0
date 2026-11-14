// metal/kernels/latent.metal - the latent MoE's own steps (Edge0 120B / 145B), after quant/units/moe.metal.
//
// A latent layer is D -> A -> U: xl[t][g] = D_g . x[t] (G groups of ELL rows, Q8_0 / Q4_0), the routed experts at the
// latent width (gate/up [I, ELL], down [ELL, I]) read the row of their group label, and y = U_g . yl.  D and U run on
// gemv.metal, the experts on moe.metal with n_embd = ELL; these kernels are the glue between them:
//
//   lat_colscale   xl[row][c] *= inv[c]: the pack's shared column scale (the gate/up rows hold A * diag(s))
//   lat_plan_rows  the plan's activation rows: token -> token * G + label[expert], so moe_gu reads the group's xl row
//                  (in place, once: it follows the layer's moe_route / moe_plan, which writes the token rows)
//   lat_group_sum  z[g][t] = sum of the weighted expert outputs of token t whose expert has label g.  U is linear, so
//                  sum_k w_k U_g(k) yl_k = sum_g U_g z_g: one GEMV a group instead of one a routed entry.

struct LatArgs {
    uint ell;      // latent width
    uint rows;     // n_tok * g: the xl rows of this window
    uint g;        // groups
    uint maxt;     // the row stride of z between groups
    uint n_tok;
    uint pad0, pad1, pad2;
};

kernel void lat_colscale(device float* xl [[buffer(0)]], device const float* inv [[buffer(1)]],
                         constant LatArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= a.rows * a.ell) return;
    xl[gid] *= inv[gid % a.ell];
}

// one thread per plan group (an expert of the window); its entries are the tokens that routed to it
kernel void lat_plan_rows(device int* plan [[buffer(0)]], device const int* labels [[buffer(1)]],
                          constant LatArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (int(gid) >= plan[P_NG]) return;
    const int lab = labels[plan[P_GEXP + gid]];
    const int c0 = plan[P_GSTART + gid], c1 = plan[P_GSTART + gid + 1];
    for (int j = c0; j < c1; ++j) plan[P_ETOK + j] = plan[P_ETOK + j] * int(a.g) + lab;
}

// z row = g * maxt + t (a group's tokens are consecutive rows: the U GEMV of group g reads them as its window)
kernel void lat_group_sum(device const float* ye [[buffer(0)]], device const int* plan [[buffer(1)]],
                          device float* z [[buffer(2)]], constant LatArgs& a [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    const uint row = gid / a.ell, c = gid % a.ell;
    if (row >= a.g * a.maxt) return;
    const uint g = row / a.maxt, t = row % a.maxt;
    float s = 0.0f;
    if (t < a.n_tok) {
        const int want = int(t * a.g + g), ne = plan[P_GSTART + plan[P_NG]];
        for (int j = 0; j < ne; ++j)
            if (plan[P_ETOK + j] == want) s += ye[j * int(a.ell) + int(c)];
    }
    z[gid] = s;
}

// lat_live: the live latent groups of a decode window, for skipping the U GEMVs of groups no routed expert uses.
// The plan holds P_NG distinct routed experts; group g is live if any of them has label g.  Writes, per group, the
// U GEMV's indirect grid (its row count, or 0 for a dead group so the dispatch does no work and reads no weights),
// and zeros a dead group's lyu rows for the live tokens, since the combine sums every group and a skipped GEMV would
// otherwise leave the previous layer's output there.  One threadgroup.
struct LatLiveArgs {
    uint g;        // groups
    uint n_tok;
    uint n_out;    // U output width (the model width N): lyu row length and the live grid's gx numerator
    uint maxt;     // lyu row stride between groups
    uint ug_gx;    // the U GEMV's threadgroup count for a live group (host passes ceil(n_out / rpt))
    uint pad0, pad1, pad2;
};
kernel void lat_live(device const int* plan [[buffer(0)]], device const int* labels [[buffer(1)]],
                     device float* lyu [[buffer(2)]], device uint* ugrid [[buffer(3)]],
                     constant LatLiveArgs& a [[buffer(4)]], uint tid [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    threadgroup atomic_uint live[16];
    for (uint g = tid; g < a.g; g += nt) atomic_store_explicit(&live[g], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ng = plan[P_NG];
    for (int gi = int(tid); gi < ng; gi += int(nt)) {
        const int lab = labels[plan[P_GEXP + gi]];
        if (lab >= 0 && lab < int(a.g)) atomic_store_explicit(&live[lab], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint live_s[16];
    for (uint g = tid; g < a.g; g += nt) {
        const uint lv = atomic_load_explicit(&live[g], memory_order_relaxed);
        live_s[g] = lv;
        ugrid[g * 3 + 0] = lv ? a.ug_gx : 0u;
        ugrid[g * 3 + 1] = 1u;
        ugrid[g * 3 + 2] = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // zero the dead groups' lyu rows for the live tokens (what moe_combine_sh reads)
    const uint total = a.g * a.n_tok * a.n_out;
    for (uint i = tid; i < total; i += nt) {
        const uint c = i % a.n_out, gt = i / a.n_out, t = gt % a.n_tok, g = gt / a.n_tok;
        if (!live_s[g]) lyu[(ulong)(g * a.maxt + t) * a.n_out + c] = 0.0f;
    }
}
