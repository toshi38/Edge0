// metal/kernels/latent_mm.metal - the latent MoE (Edge0 120B / 145B) for a PROMPT chunk, after moe_mm.metal.
//
// A chunk's latent layer runs on the prompt chain's own kernels: D as a dense GEMM (dq_f16 + gemm_f16) into
// xl[t][g] (row t * G + g), the routed experts as moe_mm.metal's grouped GEMMs at the latent width, U as one GEMM a
// group.  These kernels are the glue:
//
//   lat_mm_rows    xrow[padded row] = token * G + label[expert]: the xl row of that entry (-1 on padding)
//   lat_mm_gather  XS[padded row] = half(xl[xrow] * inv): the entry's xl row with the pack's shared column scale (the
//                  gate/up rows hold A * diag(s), so xl is multiplied by 1 / s), as the experts' fp16 operand
//   lat_mm_group   zh[g][t] = half(sum of the weighted expert outputs of token t whose expert has label g)
//   lat_mm_usum    y[t] = sum over the groups of yu[g][t]

struct LatMmArgs {
    uint g;         // groups
    uint tm;        // rows a tile
    uint ell;       // latent width
    uint n_tok;     // the chunk's rows
    uint topk;
    uint n_out;     // the model width (lat_mm_usum)
    uint pad0, pad1;
};

// one thread a tile slot (the grid covers the most tiles a chunk can have; meta[0] is this layer's count)
kernel void lat_mm_rows(device const int2* tiles [[buffer(0)]], device const uint* meta [[buffer(1)]],
                        device const int* srt_tok [[buffer(2)]], device const int* labels [[buffer(3)]],
                        device int* xrow [[buffer(4)]], constant LatMmArgs& a [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= meta[0]) return;
    const int2 tl = tiles[gid];
    const int lab = labels[tl.x];
    for (uint m = 0; m < a.tm; ++m) {
        const int t = srt_tok[tl.y + int(m)];
        xrow[tl.y + int(m)] = t >= 0 ? t * int(a.g) + lab : -1;
    }
}

// grid (rows), 256 threads (mm_gather's grid: meta[8..10])
kernel void lat_mm_gather(device const float* xl [[buffer(0)]], device const int* xrow [[buffer(1)]],
                          device const float* inv [[buffer(2)]], device half* XS [[buffer(3)]],
                          constant LatMmArgs& a [[buffer(4)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int r = xrow[tg.x];
    device half* o = XS + ulong(tg.x) * a.ell;
    for (uint k = tid; k < a.ell; k += 256) o[k] = r >= 0 ? half(xl[ulong(r) * a.ell + k] * inv[k]) : half(0.0f);
}

// one thread a (token, column): grid (n_tok * ell / 256), 256 threads; zh row = g * n_tok + t.  At most 16 groups.
kernel void lat_mm_group(device const float* YE [[buffer(0)]], device const int* pos [[buffer(1)]],
                         device const float* srt_w [[buffer(2)]], device const int* ids [[buffer(3)]],
                         device const int* labels [[buffer(4)]], device half* zh [[buffer(5)]],
                         constant LatMmArgs& a [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.ell, c = gid % a.ell;
    if (t >= a.n_tok) return;
    float s[16];
    for (uint g = 0; g < 16; ++g) s[g] = 0.0f;
    for (uint k = 0; k < a.topk; ++k) {
        const int r = pos[t * a.topk + k];
        const uint g = uint(labels[ids[t * a.topk + k]]) & 15u;
        s[g] = fma(srt_w[r], YE[ulong(r) * a.ell + c], s[g]);
    }
    for (uint g = 0; g < a.g; ++g) zh[(ulong(g) * a.n_tok + t) * a.ell + c] = half(s[g]);
}

// grid (n_tok * n_out / 256), 256 threads; yu row = g * n_tok + t
kernel void lat_mm_usum(device const float* yu [[buffer(0)]], device float* y [[buffer(1)]],
                        constant LatMmArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n_tok * a.n_out) return;
    float s = 0.0f;
    for (uint g = 0; g < a.g; ++g) s += yu[ulong(g) * a.n_tok * a.n_out + gid];
    y[gid] = s;
}
