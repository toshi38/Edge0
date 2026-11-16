// metal/tests/latent_u_parity.cpp - latent_group_u_window (one pass over a group's U for every token of a window)
// against latent_group_u (one pass a token): bit-equal outputs and a double-precision reference, for windows of 1 to 8
// tokens. --bench times both over LAYERS layers of weights, so a layer's U is not in cache when it is read.
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct LatentRoutedArgs {
    uint32_t n_tok, groups, hidden, latent, ff, topk, plan_token_entry;
    uint32_t u_row, gu_row, down_row, up_off, down_off, expert_bytes;
};
constexpr int GROUPS = 8, HIDDEN = 2560, LATENT = 1792, TOPK = 10, MAXT = 8;
}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int layers = 2;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) { bench = true; layers = 12; }
        else if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) layers = std::atoi(argv[++i]);
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "latent_routed.metal"});
    const size_t u_row = ref::row_bytes(ref::Q4_0, LATENT), layer_bytes = u_row * HIDDEN * GROUPS;
    mtl::Buffer W = cx.alloc(layer_bytes * layers);
    for (int l = 0; l < layers; ++l)
        ref::random_blocks(ref::Q4_0, W.as<uint8_t>() + l * layer_bytes, HIDDEN * GROUPS, LATENT, 11 + l, 0.02f);
    mtl::Buffer x = cx.alloc(sizeof(float) * MAXT * GROUPS * LATENT), active = cx.alloc(sizeof(uint32_t) * MAXT * GROUPS);
    mtl::Buffer y_token = cx.alloc(sizeof(float) * MAXT * GROUPS * HIDDEN), y_window = cx.alloc(y_token.bytes);
    ref::Rng r(5);
    for (int i = 0; i < MAXT * GROUPS * LATENT; ++i) x.as<float>()[i] = r.sym();
    // A token routes TOPK experts into GROUPS groups: a group is used when one of them carries its label.
    for (int t = 0; t < MAXT; ++t) {
        for (int g = 0; g < GROUPS; ++g) active.as<uint32_t>()[t * GROUPS + g] = 0;
        for (int k = 0; k < TOPK; ++k) active.as<uint32_t>()[t * GROUPS + r.next() % GROUPS] = 1;
    }
    std::vector<float> w0((size_t) HIDDEN * GROUPS * LATENT);
    ref::dequant(ref::Q4_0, W.ptr, w0.data(), w0.size());

    struct Tile { const char* token; const char* window; uint32_t rows; };
    const Tile tiles[] = {{"latent_group_u", "latent_group_u_window", 4},
                          {"latent_group_u_8", "latent_group_u_window_8", 16},
                          {"latent_group_u_4", "latent_group_u_window_4", 32}};
    int fails = 0;
    for (const Tile& tile : tiles) {
        mtl::Pipeline p_token = cx.pipeline(lib, tile.token);
        for (int nt = 1; nt <= MAXT; ++nt) {
            mtl::Pipeline p_window = cx.pipeline(lib, tile.window, mtl::Constants().i(0, nt));
            LatentRoutedArgs a{(uint32_t) nt, GROUPS, HIDDEN, LATENT, 640, TOPK, 0, (uint32_t) u_row, 0, 0, 0, 0, 0};
            auto dispatch = [&](const mtl::Pipeline& p, const mtl::Buffer& y, uint32_t slots, int layer) {
                mtl::Dispatch d;
                d.p = &p;
                d.args = {mtl::Arg::B(W, layer * layer_bytes), mtl::Arg::B(x), mtl::Arg::B(active), mtl::Arg::B(y),
                          mtl::Arg::V(a)};
                d.grid[0] = HIDDEN / tile.rows;
                d.grid[1] = slots;
                d.tg[0] = 128;
                return d;
            };
            std::memset(y_token.ptr, 0xff, y_token.bytes);
            std::memset(y_window.ptr, 0xff, y_window.bytes);
            cx.run({dispatch(p_token, y_token, nt * GROUPS, 0)});
            cx.run({dispatch(p_window, y_window, GROUPS, 0)});
            const size_t n = (size_t) nt * GROUPS * HIDDEN;
            const bool equal = !std::memcmp(y_token.ptr, y_window.ptr, n * sizeof(float));
            double se = 0, sr = 0;
            for (int s = 0; s < nt * GROUPS; ++s)
                for (int o = 0; o < HIDDEN; ++o) {
                    double ref = 0;
                    if (active.as<uint32_t>()[s]) {
                        const float* w = w0.data() + ((size_t) (s % GROUPS) * HIDDEN + o) * LATENT;
                        const float* xs = x.as<float>() + (size_t) s * LATENT;
                        for (int k = 0; k < LATENT; ++k) ref += (double) w[k] * xs[k];
                    }
                    const double e = y_window.as<float>()[(size_t) s * HIDDEN + o] - ref;
                    se += e * e;
                    sr += ref * ref;
                }
            const double rel = std::sqrt(se / sr);
            const bool pass = equal && rel < 1e-5;
            fails += !pass;
            std::printf("%-24s ntok %d: %s, rel L2 against double %.2e  %s", tile.window, nt,
                        equal ? "bit-equal" : "NOT bit-equal", rel, pass ? "PASS" : "FAIL");
            if (bench) {
                std::vector<mtl::Dispatch> per_token, per_window;
                for (int l = 0; l < layers; ++l) {
                    per_token.push_back(dispatch(p_token, y_token, nt * GROUPS, l));
                    per_window.push_back(dispatch(p_window, y_window, GROUPS, l));
                }
                cx.warm(per_token);
                double best_token = 1e9, best_window = 1e9;
                for (int rep = 0; rep < 7; ++rep) {
                    best_token = std::min(best_token, cx.run(per_token, 4) / (4 * layers));
                    best_window = std::min(best_window, cx.run(per_window, 4) / (4 * layers));
                }
                std::printf("   a layer: %6.1f us a token pass, %6.1f us a window pass (%.2fx)", best_token * 1e6,
                            best_window * 1e6, best_token / best_window);
            }
            std::printf("\n");
        }
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
