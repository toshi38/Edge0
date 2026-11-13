// metal/tests/latent_parity.cpp - one latent MoE layer (Edge0 120B / 145B) on the Metal kernels against a CPU reference.
//
//   latent_parity [--pack DIR] [--layer L] [--ntok T] [--seed S]
//
// The dispatches are the ones Window::latent_moe queues (metal/src/engine/window.cpp): D on gemv, lat_colscale, xprep,
// the plan and lat_plan_rows, moe_gu / moe_down at the latent width, lat_group_sum, U on gemv a group.  The reference
// is the layer's definition in double precision on the dequantized weights:
//   xl[t][g] = (D_g . x[t]) / s,  h = silu(gate_e . xl) * (up_e . xl),  yl = down_e . h,  out[t] = sum_k w_k U_g(e_k) . yl_k
// With --pack the layer's D/U rows, labels, column scales and expert blobs are the pack's own (experts.bin may be
// missing: the routed experts then get random blobs); without it everything is random.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, I = 640, TOPK = 10, NEXP = 512, MAXT = 8;
constexpr int P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE, P_ETOK = P_GSTART + P_MAXE + 4,
              P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE, P_INTS = P_TOKENT + P_MAXE, P_IGU = 4, P_IDN = 8;
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
struct LatArgs { uint32_t ell, rows, g, maxt, n_tok, pad0, pad1, pad2; };

bool read_at(const std::string& path, size_t off, void* dst, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const bool ok = fseeko(f, (off_t) off, SEEK_SET) == 0 && std::fread(dst, 1, n, f) == n;
    std::fclose(f);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    std::string pack;
    int layer = 0, ntok = 1, G = 8, ELL = 1792;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--ntok" && i + 1 < argc) ntok = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (uint64_t) std::atoll(argv[++i]);
        else { std::fprintf(stderr, "usage: latent_parity [--pack DIR] [--layer L] [--ntok T] [--seed S]\n"); return 2; }
    }
    if (ntok < 1 || ntok > MAXT || layer < 0 || layer > 47) return 2;
    const size_t R = (size_t) MAXT * G, plane = 4 * R * ELL;
    const size_t drow = ref::row_bytes(ref::Q8_0, N), urow = ref::row_bytes(ref::Q8_0, ELL);
    const size_t dby = (size_t) G * ELL * drow, uby = (size_t) G * N * urow;
    const size_t gu_row = ref::row_bytes(ref::Q3_K, ELL), d_row = ref::row_bytes(ref::IQ4_NL, I);
    const size_t blob = 2 * (size_t) I * gu_row + (size_t) ELL * d_row;
    ref::Rng rng(seed);

    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "moe.metal", "gemv.metal", "latent.metal"});

    // ---- the layer's weights
    mtl::Buffer LD = cx.alloc(dby + uby), LA = cx.alloc((size_t) NEXP * blob), lab = cx.alloc(4 * NEXP), linv = cx.alloc(4 * ELL);
    std::vector<float> inv((size_t) ELL, 1.0f);
    bool real_du = false, real_a = false, real_s = false;
    if (!pack.empty()) {
        real_du = read_at(pack + "/latent_dense_q8.bin", (size_t) layer * (dby + uby), LD.ptr, dby + uby);
        if (!read_at(pack + "/labels.bin", (size_t) layer * NEXP * 4, lab.ptr, 4 * NEXP)) { std::fprintf(stderr, "no labels.bin\n"); return 2; }
        std::vector<float> s((size_t) ELL);
        real_s = read_at(pack + "/scales/l" + std::to_string(layer) + ".f32", 0, s.data(), 4 * s.size());
        if (real_s) for (int c = 0; c < ELL; ++c) inv[(size_t) c] = s[(size_t) c] > 0.0f ? 1.0f / s[(size_t) c] : 1.0f;
    } else {
        for (int e = 0; e < NEXP; ++e) lab.as<int32_t>()[e] = (int) (rng.next() % (uint64_t) G);
        for (int c = 0; c < ELL; ++c) inv[(size_t) c] = 0.5f + rng.uni();
    }
    if (!real_du) {
        ref::random_blocks(ref::Q8_0, LD.as<uint8_t>(), G * ELL, N, seed + 11);
        ref::random_blocks(ref::Q8_0, LD.as<uint8_t>() + dby, G * N, ELL, seed + 12);
    }
    std::memcpy(linv.ptr, inv.data(), 4 * inv.size());

    // ---- the window: x, the routing
    std::vector<float> x((size_t) ntok * N), w((size_t) ntok * TOPK);
    std::vector<int32_t> ids((size_t) ntok * TOPK);
    for (auto& v : x) v = rng.sym() + rng.sym() + rng.sym();
    for (int t = 0; t < ntok; ++t) {
        float sum = 0;
        for (int k = 0; k < TOPK; ++k) {
            int e;
            bool dup;
            do {
                e = (int) (rng.next() % NEXP);
                if (t > 0 && k < 4) e = ids[(size_t) k];   // tokens of one window share experts
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * TOPK + j] == e;
            } while (dup);
            ids[(size_t) t * TOPK + k] = e;
            w[(size_t) t * TOPK + k] = 0.2f + rng.uni();
            sum += w[(size_t) t * TOPK + k];
        }
        for (int k = 0; k < TOPK; ++k) w[(size_t) t * TOPK + k] /= sum;
    }
    {   // the routed experts' blobs: the pack's, else random
        uint8_t* A = LA.as<uint8_t>();
        real_a = !pack.empty();
        for (int i = 0; i < ntok * TOPK; ++i) {
            const int e = ids[(size_t) i];
            if (real_a && !read_at(pack + "/experts.bin", ((size_t) layer * NEXP + e) * blob, A + (size_t) e * blob, blob)) real_a = false;
            if (!real_a) {
                ref::random_blocks(ref::Q3_K, A + (size_t) e * blob, 2 * I, ELL, seed + 100 + e);
                ref::random_blocks(ref::IQ4_NL, A + (size_t) e * blob + 2 * (size_t) I * gu_row, ELL, I, seed + 900 + e);
            }
        }
    }
    std::printf("layer %d, %d token(s): D/U %s, experts %s, column scale %s\n", layer, ntok, real_du ? "pack" : "random",
                real_a ? "pack" : "random", real_s ? "pack" : (pack.empty() ? "random" : "none"));

    // ---- the Metal layer
    mtl::Buffer bx = cx.alloc(4 * (size_t) MAXT * N), XF = cx.alloc(4 * ((size_t) 4 * MAXT * N + (size_t) MAXT * N / 16));
    mtl::Buffer lxl = cx.alloc(4 * R * ELL), lz = cx.alloc(4 * R * ELL), lyu = cx.alloc(4 * R * N);
    mtl::Buffer LXb = cx.alloc(4 * (4 * R * ELL + R * ELL / 16)), LZb = cx.alloc(4 * (4 * R * ELL + R * ELL / 16));
    mtl::Buffer bids = cx.alloc(4 * MAXT * TOPK), bw = cx.alloc(4 * MAXT * TOPK), plan = cx.alloc(4 * P_INTS);
    mtl::Buffer h = cx.alloc(4 * (size_t) MAXT * TOPK * I), ye = cx.alloc(4 * (size_t) MAXT * TOPK * N);
    std::memcpy(bx.ptr, x.data(), 4 * x.size());
    std::memcpy(bids.ptr, ids.data(), 4 * ids.size());
    std::memcpy(bw.ptr, w.data(), 4 * w.size());

    const mtl::Pipeline pprep = cx.pipeline(lib, "xprep"), pplan = cx.pipeline(lib, "moe_plan"),
                        pcol = cx.pipeline(lib, "lat_colscale"), prow = cx.pipeline(lib, "lat_plan_rows"),
                        psum = cx.pipeline(lib, "lat_group_sum"),
                        pgemv = cx.pipeline(lib, "gemv_8_16_1_4_1", mtl::Constants().i(0, ntok)),
                        pgu = cx.pipeline(lib, "moe_gu_11_32_1_4"), pdn = cx.pipeline(lib, "moe_down_20_4_2_4");
    const LatArgs la{(uint32_t) ELL, (uint32_t) (ntok * G), (uint32_t) G, (uint32_t) MAXT, (uint32_t) ntok, 0, 0, 0};
    MoeArgs a{};
    a.gu_row = (uint32_t) gu_row; a.d_row = (uint32_t) d_row;
    a.gate_off = 0; a.up_off = (uint64_t) I * gu_row; a.down_off = 2 * a.up_off;
    a.gu_stride = a.d_stride = blob;
    a.n_embd = ELL; a.n_ff = I; a.n_tok = ntok; a.topk = TOPK; a.n_expert = NEXP;
    a.gx_gu = I / 4; a.gx_dn = ELL / 64;
    auto prep = [&](const mtl::Buffer& src, const mtl::Buffer& dst, uint32_t n, uint32_t nt) {
        mtl::Dispatch d;
        d.p = &pprep;
        d.args = {mtl::Arg::B(src), mtl::Arg::B(dst), mtl::Arg::V(n), mtl::Arg::V(nt)};
        d.tg[0] = 256;
        d.grid[0] = (n / 16 * nt + 255) / 256;
        return d;
    };
    auto gemv = [&](size_t w_off, int n_in, int n_out, const mtl::Buffer& xp, size_t pl, size_t row0, const mtl::Buffer& y,
                    size_t y_off, size_t y_stride) {
        GemvArgs ga{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ntok, (uint32_t) ref::row_bytes(ref::Q8_0, n_in), (uint32_t) n_in,
                    (uint32_t) y_stride};
        mtl::Dispatch d;
        d.p = &pgemv;
        d.args = {mtl::Arg::B(LD, w_off), mtl::Arg::B(xp, row0 * (size_t) n_in * 4),
                  mtl::Arg::B(xp, pl * 4 + row0 * (size_t) (n_in / 16) * 4), mtl::Arg::B(y, y_off), mtl::Arg::V(ga)};
        d.grid[0] = (n_out + 7) / 8;
        d.tg[0] = 128;
        return d;
    };
    std::vector<mtl::Dispatch> ds;
    ds.push_back(prep(bx, XF, N, MAXT));
    ds.push_back(gemv(0, N, G * ELL, XF, (size_t) 4 * MAXT * N, 0, lxl, 0, (size_t) G * ELL));
    {
        mtl::Dispatch d;
        d.p = &pcol;
        d.args = {mtl::Arg::B(lxl), mtl::Arg::B(linv), mtl::Arg::V(la)};
        d.tg[0] = 256;
        d.grid[0] = (uint32_t) ((ntok * G * ELL + 255) / 256);
        ds.push_back(d);
    }
    ds.push_back(prep(lxl, LXb, ELL, (uint32_t) R));
    {
        mtl::Dispatch d;
        d.p = &pplan;
        d.args = {mtl::Arg::B(bids), mtl::Arg::B(bw), mtl::Arg::B(plan), mtl::Arg::V(a)};
        d.tg[0] = 512;
        ds.push_back(d);
        mtl::Dispatch r;
        r.p = &prow;
        r.args = {mtl::Arg::B(plan), mtl::Arg::B(lab), mtl::Arg::V(la)};
        r.tg[0] = P_MAXE;
        ds.push_back(r);
    }
    {
        mtl::Dispatch g;
        g.p = &pgu;
        g.args = {mtl::Arg::B(LA), mtl::Arg::B(LXb, plane * 3), mtl::Arg::B(h), mtl::Arg::B(plan), mtl::Arg::V(a),
                  mtl::Arg::B(LXb, plane * 4)};
        g.tg[0] = 128;
        g.indirect = &plan;
        g.ind_off = P_IGU * 4;
        ds.push_back(g);
        mtl::Dispatch d;
        d.p = &pdn;
        d.args = {mtl::Arg::B(LA), mtl::Arg::B(h), mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::V(a), mtl::Arg::B(h)};
        d.tg[0] = 128;
        d.indirect = &plan;
        d.ind_off = P_IDN * 4;
        ds.push_back(d);
    }
    {
        mtl::Dispatch d;
        d.p = &psum;
        d.args = {mtl::Arg::B(ye), mtl::Arg::B(plan), mtl::Arg::B(lz), mtl::Arg::V(la)};
        d.tg[0] = 256;
        d.grid[0] = (uint32_t) ((R * ELL + 255) / 256);
        ds.push_back(d);
    }
    ds.push_back(prep(lz, LZb, ELL, (uint32_t) R));
    for (int g = 0; g < G; ++g)
        ds.push_back(gemv(dby + (size_t) g * N * urow, ELL, N, LZb, plane, (size_t) g * MAXT, lyu, (size_t) g * MAXT * N * 4, N));
    cx.run(ds, 1, false, true);
    // the layer's time: the same dispatches repeated (warm weights), wall clock a repetition
    const auto t0 = std::chrono::steady_clock::now();
    const int reps = 20;
    cx.run(ds, reps, false, true);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::vector<float> got((size_t) ntok * N, 0.0f);
    for (int t = 0; t < ntok; ++t)
        for (int g = 0; g < G; ++g)
            for (int c = 0; c < N; ++c) got[(size_t) t * N + c] += lyu.as<float>()[((size_t) g * MAXT + t) * N + c];

    // ---- the reference
    std::vector<double> want((size_t) ntok * N, 0.0), xl((size_t) ntok * G * ELL), xl_gpu_err(2, 0.0);
    std::vector<float> row((size_t) N);
    const uint8_t *D = LD.as<uint8_t>(), *U = D + dby, *A = LA.as<uint8_t>();
    for (int r = 0; r < G * ELL; ++r) {
        ref::dequant(ref::Q8_0, D + (size_t) r * drow, row.data(), N);
        for (int t = 0; t < ntok; ++t) {
            double s = 0;
            for (int c = 0; c < N; ++c) s += (double) row[(size_t) c] * x[(size_t) t * N + c];
            xl[((size_t) t * G) * ELL + r] = s * inv[(size_t) (r % ELL)];
        }
    }
    for (size_t i = 0; i < xl.size(); ++i) {
        const double d = lxl.as<float>()[i] - xl[i];
        xl_gpu_err[0] += d * d; xl_gpu_err[1] += xl[i] * xl[i];
    }
    std::vector<float> wg((size_t) I * ELL), wu((size_t) I * ELL), wd((size_t) ELL * I), ur((size_t) ELL);
    std::vector<double> hh((size_t) I), yl((size_t) ELL);
    for (int i = 0; i < ntok * TOPK; ++i) {
        const int e = ids[(size_t) i], t = i / TOPK, g = lab.as<int32_t>()[e];
        const uint8_t* b = A + (size_t) e * blob;
        ref::dequant(ref::Q3_K, b, wg.data(), wg.size());
        ref::dequant(ref::Q3_K, b + (size_t) I * gu_row, wu.data(), wu.size());
        ref::dequant(ref::IQ4_NL, b + 2 * (size_t) I * gu_row, wd.data(), wd.size());
        const double* xr = xl.data() + ((size_t) t * G + g) * ELL;
        for (int r = 0; r < I; ++r) {
            double sg = 0, su = 0;
            for (int c = 0; c < ELL; ++c) { sg += (double) wg[(size_t) r * ELL + c] * xr[c]; su += (double) wu[(size_t) r * ELL + c] * xr[c]; }
            hh[(size_t) r] = sg / (1.0 + std::exp(-sg)) * su;
        }
        for (int r = 0; r < ELL; ++r) {
            double s = 0;
            for (int c = 0; c < I; ++c) s += (double) wd[(size_t) r * I + c] * hh[(size_t) c];
            yl[(size_t) r] = s;
        }
        for (int r = 0; r < N; ++r) {
            ref::dequant(ref::Q8_0, U + ((size_t) g * N + r) * urow, ur.data(), ELL);
            double s = 0;
            for (int c = 0; c < ELL; ++c) s += (double) ur[(size_t) c] * yl[(size_t) c];
            want[(size_t) t * N + r] += (double) w[(size_t) i] * s;
        }
    }
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = got[i] - want[i];
        num += d * d; den += want[i] * want[i];
        mx = std::fmax(mx, std::fabs(d));
    }
    const double rel = std::sqrt(num / (den + 1e-30)), relx = std::sqrt(xl_gpu_err[0] / (xl_gpu_err[1] + 1e-30));
    std::printf("xl   rel L2 %.3e\nout  rel L2 %.3e  max abs %.3e  (|out| rms %.3e)\nlayer %.2f ms for %d token(s): %.2f ms a token, %.0f ms a token over 48 layers\n", relx, rel, mx,
                std::sqrt(den / want.size()), ms, ntok, ms / ntok, 48 * ms / ntok);
    const bool ok = rel < 2e-3 && relx < 1e-4 && den > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
