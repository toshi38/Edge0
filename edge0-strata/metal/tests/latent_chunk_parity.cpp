// metal/tests/latent_chunk_parity.cpp - one latent MoE layer (Edge0 120B / 145B) of a PROMPT chunk on the prompt chain's
// kernels against a CPU reference.
//
//   latent_chunk_parity [--pack DIR] [--layer L] [--ntok T] [--check C] [--seed S] [--gu M_N_K_S] [--dn M_N_K_S]
//
// --gu / --dn name the gate/up and down GEMM tiles (rows, outputs, K chunk, simdgroups); both must have the same rows.
//
// The dispatches are the ones the latent branch of Window::pc_chunk queues (metal/src/engine/prompt_chain.inc): D as
// dq_f16 + gemm_f16, the plan (mm_clear / mm_hist / mm_scan / mm_place), lat_mm_rows, lat_mm_gather, mm_gu2 and
// mm_down2 at the latent width, lat_mm_group, U as dq_f16 + gemm_f16 a group, lat_mm_usum.  The reference is the layer's
// definition in double precision on the dequantized weights (latent_parity.cpp's), for the first C tokens of the chunk.
// The operands are fp16, so the tolerance is the prompt chain's, not the decode window's.
// With --pack the layer's D/U rows, labels, column scales and expert blobs are the pack's own; without it random.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, I = 640, TOPK = 10, NEXP = 512;
struct DqArgs { uint32_t n_in, n_out, row_bytes, perm; };
struct GemmArgs { int M, N, K, pad; };
struct PlanArgs { uint32_t n_ent, topk, n_expert, tm, n_blocks; };
struct MmArgs { uint64_t gate_off, up_off, down_off, gu_stride, d_stride; uint32_t gu_row, d_row, n_embd, n_ff; };
struct LatArgs { uint32_t g, tm, ell, n_tok, topk, n_out, pad0, pad1; };

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
    int layer = 0, T = 256, check = 16, G = 8, ELL = 1792;
    uint64_t seed = 1;
    std::string gu_tile = "32_32_128_4", dn_tile = "32_64_32_2";   // the engine's (prompt_chain.inc)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--ntok" && i + 1 < argc) T = std::atoi(argv[++i]);
        else if (a == "--check" && i + 1 < argc) check = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = (uint64_t) std::atoll(argv[++i]);
        else if (a == "--gu" && i + 1 < argc) gu_tile = argv[++i];
        else if (a == "--dn" && i + 1 < argc) dn_tile = argv[++i];
        else { std::fprintf(stderr, "usage: latent_chunk_parity [--pack DIR] [--layer L] [--ntok T] [--check C] [--seed S] [--gu M_N_K_S] [--dn M_N_K_S]\n"); return 2; }
    }
    int gt[4] = {0, 0, 0, 0}, dt[4] = {0, 0, 0, 0};
    if (std::sscanf(gu_tile.c_str(), "%d_%d_%d_%d", &gt[0], &gt[1], &gt[2], &gt[3]) != 4 ||
        std::sscanf(dn_tile.c_str(), "%d_%d_%d_%d", &dt[0], &dt[1], &dt[2], &dt[3]) != 4 || gt[0] != dt[0]) return 2;
    const int TM = gt[0], GU_TN = gt[1], DN_TN = dt[1];
    if (T < 1 || T > 8192 || layer < 0 || layer > 47) return 2;
    check = std::max(1, std::min(check, T));
    const size_t drow = ref::row_bytes(ref::Q8_0, N), urow = ref::row_bytes(ref::Q8_0, ELL);
    const size_t dby = (size_t) G * ELL * drow, uby = (size_t) G * N * urow;
    const size_t gu_row = ref::row_bytes(ref::Q3_K, ELL), d_row = ref::row_bytes(ref::IQ4_NL, I);
    const size_t blob = 2 * (size_t) I * gu_row + (size_t) ELL * d_row;
    ref::Rng rng(seed);

    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "mma.metal", "dense_mm.metal", "moe_mm.metal", "latent_mm.metal"});

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
        real_a = read_at(pack + "/experts.bin", (size_t) layer * NEXP * blob, LA.ptr, (size_t) NEXP * blob);
    } else {
        for (int e = 0; e < NEXP; ++e) lab.as<int32_t>()[e] = (int) (rng.next() % (uint64_t) G);
        for (int c = 0; c < ELL; ++c) inv[(size_t) c] = 0.5f + rng.uni();
    }
    if (!real_du) {
        ref::random_blocks(ref::Q8_0, LD.as<uint8_t>(), G * ELL, N, seed + 11);
        ref::random_blocks(ref::Q8_0, LD.as<uint8_t>() + dby, G * N, ELL, seed + 12);
    }
    if (!real_a)
        for (int e = 0; e < NEXP; ++e) {
            ref::random_blocks(ref::Q3_K, LA.as<uint8_t>() + (size_t) e * blob, 2 * I, ELL, seed + 100 + e);
            ref::random_blocks(ref::IQ4_NL, LA.as<uint8_t>() + (size_t) e * blob + 2 * (size_t) I * gu_row, ELL, I, seed + 900 + e);
        }
    std::memcpy(linv.ptr, inv.data(), 4 * inv.size());

    // ---- the chunk: x, the routing (experts drawn with a skew, as a real chunk's are)
    std::vector<float> x((size_t) T * N), w((size_t) T * TOPK);
    std::vector<int32_t> ids((size_t) T * TOPK);
    for (auto& v : x) v = rng.sym() + rng.sym() + rng.sym();
    for (int t = 0; t < T; ++t) {
        float sum = 0;
        for (int k = 0; k < TOPK; ++k) {
            int e;
            bool dup;
            do {
                const double u = rng.uni();
                e = (int) (u * u * NEXP) % NEXP;
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * TOPK + j] == e;
            } while (dup);
            ids[(size_t) t * TOPK + k] = e;
            w[(size_t) t * TOPK + k] = 0.2f + rng.uni();
            sum += w[(size_t) t * TOPK + k];
        }
        for (int k = 0; k < TOPK; ++k) w[(size_t) t * TOPK + k] /= sum;
    }
    std::printf("layer %d, chunk of %d rows: D/U %s, experts %s, column scale %s\n", layer, T, real_du ? "pack" : "random",
                real_a ? "pack" : "random", real_s ? "pack" : (pack.empty() ? "random" : "none"));

    // ---- the Metal layer
    const int nent = T * TOPK, nblk = (nent + 1023) / 1024, maxrows = nent + NEXP * TM;
    const size_t MT = (size_t) T, mr = (size_t) maxrows;
    mtl::Buffer bx = cx.alloc(4 * MT * N), xh = cx.alloc(2 * MT * N), LWD = cx.alloc(2 * (size_t) G * ELL * N),
                LXL = cx.alloc(4 * MT * G * ELL), lrow = cx.alloc(4 * mr), LZH = cx.alloc(2 * (size_t) G * MT * ELL),
                LWU = cx.alloc(2 * (size_t) N * ELL), LYU = cx.alloc(4 * (size_t) G * MT * N), yr = cx.alloc(4 * MT * N),
                bids = cx.alloc(4 * MT * TOPK), bw = cx.alloc(4 * MT * TOPK), bcnt = cx.alloc(4 * (size_t) nblk * NEXP),
                boff = cx.alloc(4 * (size_t) nblk * NEXP), eoff = cx.alloc(4 * NEXP), tiles = cx.alloc(8 * (mr / TM + 1)),
                meta = cx.alloc(64), stok = cx.alloc(4 * mr), sw = cx.alloc(4 * mr), pos = cx.alloc(4 * MT * TOPK),
                XS = cx.alloc(2 * mr * ELL), HS = cx.alloc(2 * mr * I), YE = cx.alloc(4 * mr * ELL);
    std::memcpy(bx.ptr, x.data(), 4 * x.size());
    std::memcpy(bids.ptr, ids.data(), 4 * ids.size());
    std::memcpy(bw.ptr, w.data(), 4 * w.size());
    meta.as<uint32_t>()[2] = I / GU_TN;
    meta.as<uint32_t>()[5] = ELL / DN_TN;

    const mtl::Pipeline pact = cx.pipeline(lib, "act_f16"), pdq = cx.pipeline(lib, "dq_f16_8"), pgemm = cx.pipeline(lib, "gemm_f16_64_64_4"),
                        pclear = cx.pipeline(lib, "mm_clear"), phist = cx.pipeline(lib, "mm_hist"),
                        pscan = cx.pipeline(lib, "mm_scan"), pplace = cx.pipeline(lib, "mm_place"), prows = cx.pipeline(lib, "lat_mm_rows"),
                        pgather = cx.pipeline(lib, "lat_mm_gather"),
                        pgu = cx.pipeline(lib, "mm_gu2_11_" + gu_tile, mtl::Constants().i(5, 0)),
                        pdn = cx.pipeline(lib, "mm_down2_20_" + dn_tile), pgroup = cx.pipeline(lib, "lat_mm_group"),
                        pusum = cx.pipeline(lib, "lat_mm_usum");
    std::vector<mtl::Dispatch> ds;
    auto D = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> a, uint32_t gx, uint32_t gy, uint32_t tg) {
        mtl::Dispatch d;
        d.p = &p;
        d.args = std::move(a);
        d.grid[0] = gx; d.grid[1] = gy;
        d.tg[0] = tg;
        ds.push_back(std::move(d));
    };
    auto IND = [&](const mtl::Pipeline& p, std::vector<mtl::Arg> a, uint32_t tg, size_t ind_off) {
        mtl::Dispatch d;
        d.p = &p;
        d.args = std::move(a);
        d.tg[0] = tg;
        d.indirect = &meta;
        d.ind_off = ind_off;
        ds.push_back(std::move(d));
    };
    auto dq = [&](size_t off, int n_in, int n_out, const mtl::Buffer& out) {
        DqArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ref::row_bytes(ref::Q8_0, n_in), 0};
        D(pdq, {mtl::Arg::B(LD, off), mtl::Arg::B(out), mtl::Arg::V(a)}, (n_in / 32 + 63) / 64, n_out, 64);
    };
    const LatArgs la{(uint32_t) G, (uint32_t) TM, (uint32_t) ELL, (uint32_t) T, TOPK, N, 0, 0};
    const PlanArgs pa{(uint32_t) nent, TOPK, NEXP, (uint32_t) TM, (uint32_t) nblk};
    MmArgs ma{0, (uint64_t) I * gu_row, 2 * (uint64_t) I * gu_row, blob, blob, (uint32_t) gu_row, (uint32_t) d_row, (uint32_t) ELL, I};
    const uint32_t nx = (uint32_t) (MT * N), nclr = (uint32_t) maxrows;
    D(pact, {mtl::Arg::B(bx), mtl::Arg::B(xh), mtl::Arg::V(nx)}, (nx + 255) / 256, 1, 256);
    const size_t first_layer_dispatch = ds.size();
    dq(0, N, G * ELL, LWD);
    {
        GemmArgs g{T, G * ELL, N, 0};
        D(pgemm, {mtl::Arg::B(xh), mtl::Arg::B(LWD), mtl::Arg::B(LXL), mtl::Arg::V(g)}, (G * ELL + 63) / 64, (T + 63) / 64, 128);
    }
    D(pclear, {mtl::Arg::B(stok), mtl::Arg::V(nclr)}, (maxrows + 255) / 256, 1, 256);
    D(phist, {mtl::Arg::B(bids), mtl::Arg::B(bcnt), mtl::Arg::V(pa)}, nblk, 1, 1024);
    D(pscan, {mtl::Arg::B(bcnt), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(tiles), mtl::Arg::B(meta), mtl::Arg::V(pa)}, 1, 1, NEXP);
    D(pplace, {mtl::Arg::B(bids), mtl::Arg::B(bw), mtl::Arg::B(boff), mtl::Arg::B(eoff), mtl::Arg::B(stok), mtl::Arg::B(sw),
               mtl::Arg::B(pos), mtl::Arg::V(pa)}, nblk, 1, 1024);
    D(prows, {mtl::Arg::B(tiles), mtl::Arg::B(meta), mtl::Arg::B(stok), mtl::Arg::B(lab), mtl::Arg::B(lrow), mtl::Arg::V(la)},
      (maxrows / TM + 255) / 256, 1, 256);
    IND(pgather, {mtl::Arg::B(LXL), mtl::Arg::B(lrow), mtl::Arg::B(linv), mtl::Arg::B(XS), mtl::Arg::V(la)}, 256, 8 * 4);
    IND(pgu, {mtl::Arg::B(LA), mtl::Arg::B(XS), mtl::Arg::B(HS), mtl::Arg::B(tiles), mtl::Arg::V(ma)}, 32 * (uint32_t) gt[3], 2 * 4);
    IND(pdn, {mtl::Arg::B(LA), mtl::Arg::B(HS), mtl::Arg::B(YE), mtl::Arg::B(tiles), mtl::Arg::V(ma)}, 32 * (uint32_t) dt[3], 5 * 4);
    D(pgroup, {mtl::Arg::B(YE), mtl::Arg::B(pos), mtl::Arg::B(sw), mtl::Arg::B(bids), mtl::Arg::B(lab), mtl::Arg::B(LZH), mtl::Arg::V(la)},
      (uint32_t) ((MT * ELL + 255) / 256), 1, 256);
    for (int g = 0; g < G; ++g) {
        dq(dby + (size_t) g * N * urow, ELL, N, LWU);
        GemmArgs ga{T, N, ELL, 0};
        D(pgemm, {mtl::Arg::B(LZH, (size_t) g * MT * ELL * 2), mtl::Arg::B(LWU), mtl::Arg::B(LYU, (size_t) g * MT * N * 4), mtl::Arg::V(ga)},
          (N + 63) / 64, (T + 63) / 64, 128);
    }
    D(pusum, {mtl::Arg::B(LYU), mtl::Arg::B(yr), mtl::Arg::V(la)}, (uint32_t) ((MT * N + 255) / 256), 1, 256);
    cx.run(ds, 1, false, true);
    // the layer's time: the same dispatches repeated (warm weights), wall clock a repetition; then each stage on its own
    const int reps = 5;
    auto timed = [&](const std::vector<mtl::Dispatch>& v) {
        const auto t0 = std::chrono::steady_clock::now();
        cx.run(v, reps, false, true);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const std::vector<mtl::Dispatch> layer_ds(ds.begin() + (long) first_layer_dispatch, ds.end());
    const double ms = timed(layer_ds);
    const char* names[] = {"D dq", "D gemm", "plan", "plan", "plan", "plan", "rows", "gather", "gate/up", "down", "group sum"};
    double st[5] = {0, 0, 0, 0, 0};   // D, plan + glue, experts, U, other
    for (size_t i = 0; i < layer_ds.size(); ++i) {
        const double m = timed({layer_ds[i]});
        const int k = i < 2 ? 0 : i == 8 || i == 9 ? 2 : i >= 11 && i + 1 < layer_ds.size() ? 3 : 1;
        st[k] += m;
        if (i < 11 && std::getenv("LCP_VERBOSE")) std::printf("  %-10s %.2f ms\n", names[i], m);
    }

    // ---- the reference, for the first `check` tokens
    const float* got = yr.as<float>();
    std::vector<double> want((size_t) check * N, 0.0), xl((size_t) check * G * ELL), xe(2, 0.0);
    std::vector<float> row((size_t) N);
    const uint8_t *Dw = LD.as<uint8_t>(), *U = Dw + dby, *A = LA.as<uint8_t>();
    for (int r = 0; r < G * ELL; ++r) {
        ref::dequant(ref::Q8_0, Dw + (size_t) r * drow, row.data(), N);
        for (int t = 0; t < check; ++t) {
            double s = 0;
            for (int c = 0; c < N; ++c) s += (double) row[(size_t) c] * x[(size_t) t * N + c];
            xl[((size_t) t * G) * ELL + r] = s;
        }
    }
    for (size_t i = 0; i < xl.size(); ++i) {   // the kernels' xl is before the column scale
        const double d = LXL.as<float>()[i] - xl[i];
        xe[0] += d * d; xe[1] += xl[i] * xl[i];
        xl[i] *= inv[i % (size_t) ELL];
    }
    std::vector<float> wg((size_t) I * ELL), wu((size_t) I * ELL), wd((size_t) ELL * I), um((size_t) N * ELL);
    std::vector<double> hh((size_t) I), z((size_t) check * G * ELL, 0.0);
    for (int i = 0; i < check * TOPK; ++i) {
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
        double* zr = z.data() + ((size_t) t * G + g) * ELL;
        for (int r = 0; r < ELL; ++r) {
            double s = 0;
            for (int c = 0; c < I; ++c) s += (double) wd[(size_t) r * I + c] * hh[(size_t) c];
            zr[r] += (double) w[(size_t) i] * s;
        }
    }
    for (int g = 0; g < G; ++g) {   // U is linear: applied to the group sums, as the definition's sum over the entries
        ref::dequant(ref::Q8_0, U + (size_t) g * N * urow, um.data(), um.size());
        for (int t = 0; t < check; ++t) {
            const double* zr = z.data() + ((size_t) t * G + g) * ELL;
            for (int r = 0; r < N; ++r) {
                double s = 0;
                for (int c = 0; c < ELL; ++c) s += (double) um[(size_t) r * ELL + c] * zr[c];
                want[(size_t) t * N + r] += s;
            }
        }
    }
    double num = 0, den = 0, mx = 0;
    bool finite = true;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = got[i] - want[i];
        finite &= std::isfinite(got[i]);
        num += d * d; den += want[i] * want[i];
        mx = std::fmax(mx, std::fabs(d));
    }
    const double rel = std::sqrt(num / (den + 1e-30)), relx = std::sqrt(xe[0] / (xe[1] + 1e-30));
    std::printf("xl   rel L2 %.3e\nout  rel L2 %.3e  max abs %.3e  (|out| rms %.3e), %d of %d tokens checked\n", relx, rel, mx,
                std::sqrt(den / want.size()), check, T);
    std::printf("layer %.1f ms for %d rows (D %.1f, experts %.1f, U %.1f, plan and glue %.1f): %.0f tok/s if the 48 latent layers were all\n",
                ms, T, st[0], st[2], st[3], st[1], 1000.0 * T / (48 * ms));
    const bool ok = finite && rel < 5e-3 && relx < 2e-3 && den > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
