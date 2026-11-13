// metal/src/engine/stubs_edge0.cpp - the kernel-library entry points the Edge0 latent-MoE sources reference and Metal does not
// implement yet. Same job and shape as stubs.cpp: each stops the engine with its name (metal/src/engine/missing.cpp). A symbol
// that gets a Metal implementation is deleted from here. latent_fused_available answers "no": the fused latent kernels are CUDA only.

#include <cstddef>
#include <cstdint>
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/latent.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_q3k.hpp"
#include "strata/prefill/lat_q3k_fused.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels { void copy_bytes_from_mapped(void* p0, void const* p1, long long p2, void* p3) { strata::metal_engine::missing("strata::kernels::copy_bytes_from_mapped"); } }
namespace strata::kernels { void doorbell_publish_value(float const* p0, int const* p1, float const* p2, long long p3, long long p4, float* p5, int* p6, float* p7, unsigned int* p8, unsigned int p9, void* p10) { strata::metal_engine::missing("strata::kernels::doorbell_publish_value"); } }
namespace strata::kernels { void iq_mmvq(int p0, void const* p1, void const* p2, float* p3, int p4, int p5, int p6, void* p7) { strata::metal_engine::missing("strata::kernels::iq_mmvq"); } }
namespace strata::kernels { void lat_colscale_rows(float* p0, long long p1, int p2, float const* p3, void* p4) { strata::metal_engine::missing("strata::kernels::lat_colscale_rows"); } }
namespace strata::kernels { const float* lat_gu_colscale_dev(int p0, int p1) { strata::metal_engine::missing("strata::kernels::lat_gu_colscale_dev"); return nullptr; } }
namespace strata::kernels { void latent_plan_rows(int* p0, int const* p1, int const* p2, int const* p3, int const* p4, long long p5, int p6, int p7, int p8, void* p9) { strata::metal_engine::missing("strata::kernels::latent_plan_rows"); } }
namespace strata::kernels { void latent_u_grouped(unsigned char const* p0, unsigned char const* p1, int const* p2, int const* p3, long long p4, long long p5, int p6, int p7, float* p8, void* p9, int p10) { strata::metal_engine::missing("strata::kernels::latent_u_grouped"); } }
namespace strata::kernels { void latent_u_matvec(unsigned char const* p0, unsigned char const* p1, int const* p2, int const* p3, int const* p4, int const* p5, long long p6, long long p7, int p8, int p9, int p10, float* p11, void* p12, int p13) { strata::metal_engine::missing("strata::kernels::latent_u_matvec"); } }
namespace strata::kernels { void moe_hit_add_mapped(float* p0, float const* p1, float const volatile* p2, int const* p3, int const* p4, long long p5, long long p6, unsigned int const volatile* p7, unsigned int p8, void* p9) { strata::metal_engine::missing("strata::kernels::moe_hit_add_mapped"); } }
namespace strata::kernels { void moe_hit_set(float* p0, float const* p1, int const* p2, int const* p3, long long p4, long long p5, void* p6) { strata::metal_engine::missing("strata::kernels::moe_hit_set"); } }
namespace strata::kernels { void quantize_q8_0_rows(float const* p0, long long p1, long long p2, void* p3, void* p4) { strata::metal_engine::missing("strata::kernels::quantize_q8_0_rows"); } }
namespace strata::prefill::fused { void experts_latent(strata::prefill::fused::WideBatch const& p0, strata::prefill::fused::LatentGeom const& p1, int p2, long long p3, void const* p4, void const* p5, int const* p6, void* p7, float* p8, void* p9) { strata::metal_engine::missing("strata::prefill::fused::experts_latent"); } }
namespace strata::prefill::fused { void group_latent(int const* p0, int const* p1, int p2, void* p3, unsigned long p4) { strata::metal_engine::missing("strata::prefill::fused::group_latent"); } }
namespace strata::prefill::fused { void latent_dense_d(void const* p0, void const* p1, long long p2, int p3, int p4, long long p5, void* p6, void* p7, int p8) { strata::metal_engine::missing("strata::prefill::fused::latent_dense_d"); } }
namespace strata::prefill::fused { void latent_dense_u(void const* p0, void const* p1, int const* p2, int const* p3, int p4, long long p5, long long p6, int p7, int p8, float* p9, void* p10, int p11) { strata::metal_engine::missing("strata::prefill::fused::latent_dense_u"); } }
namespace strata::prefill::fused { void latent_dense_u_tables(int const* p0, int const* p1, long long p2, int p3, int* p4, int* p5, int* p6, void* p7) { strata::metal_engine::missing("strata::prefill::fused::latent_dense_u_tables"); } }
namespace strata::prefill::fused { bool latent_fused_available() { return false; } }
namespace strata::prefill::fused { size_t latent_group_bytes(long long p0, int p1) { strata::metal_engine::missing("strata::prefill::fused::latent_group_bytes"); return {}; } }
namespace strata::prefill::fused { size_t latent_h_off(long long p0, int p1) { strata::metal_engine::missing("strata::prefill::fused::latent_h_off"); return {}; } }
namespace strata::prefill::fused { size_t latent_scratch_bytes(long long p0, int p1, long long p2) { strata::metal_engine::missing("strata::prefill::fused::latent_scratch_bytes"); return {}; } }
namespace strata::prefill::fused_q3k { void experts_latent_q3k(strata::prefill::fused::WideBatch const& p0, strata::prefill::fused::LatentGeom const& p1, int p2, long long p3, void const* p4, void const* p5, int const* p6, void* p7, float* p8, void* p9) { strata::metal_engine::missing("strata::prefill::fused_q3k::experts_latent_q3k"); } }
namespace strata::prefill::fused_q3k { void latent_dense_d(void const* p0, void const* p1, long long p2, int p3, int p4, long long p5, void* p6, void* p7, int p8, float const* p9) { strata::metal_engine::missing("strata::prefill::fused_q3k::latent_dense_d"); } }
namespace strata::prefill { void lat_q3k_fused(unsigned char const* const* p0, int const* p1, int p2, void const* p3, void* p4, float* p5, long long p6, void* p7, int const* p8, int const* p9, int const* p10) { strata::metal_engine::missing("strata::prefill::lat_q3k_fused"); } }
