#include "hc-sc-comb.cuh"

#include <cmath>

static constexpr int32_t GGML_HSCB_TAG = 0x51534342; // "HSCB"

static bool ggml_cuda_hscb_walk_bcast(const ggml_tensor * add, const ggml_tensor ** residual,
        const ggml_tensor ** block, const ggml_tensor ** inject, int64_t * hc_out) {
    if (!add || add->op != GGML_OP_ADD || add->src[1]->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * mul = add->src[1];
    const ggml_tensor * b_r = mul->src[0];
    const ggml_tensor * w_r = mul->src[1];
    if (!b_r || b_r->op != GGML_OP_RESHAPE || !w_r || w_r->op != GGML_OP_RESHAPE) {
        return false;
    }
    const ggml_tensor * block_2d = b_r->src[0];
    const ggml_tensor * w_scale2 = w_r->src[0];
    if (!w_scale2 || w_scale2->op != GGML_OP_SCALE) {
        return false;
    }
    const float scale2 = ((const float *) w_scale2->op_params)[0];
    if (std::fabs(scale2 - 2.0f) > 1e-5f) {
        return false;
    }
    const ggml_tensor * sig = w_scale2->src[0];
    if (!sig || sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * inj_scale = sig->src[0];
    if (!inj_scale || inj_scale->op != GGML_OP_SCALE) {
        return false;
    }
    const float inv_hc = ((const float *) inj_scale->op_params)[0];
    if (inv_hc <= 0.0f) {
        return false;
    }
    const int64_t hc = (int64_t) std::lround(1.0f / inv_hc);
    if (hc <= 0 || hc > 64) {
        return false;
    }
    *residual = add->src[0];
    *block    = block_2d;
    *inject   = inj_scale->src[0];
    *hc_out   = hc;
    return b_r->ne[1] == 1 && w_r->ne[0] == 1 && w_r->ne[1] == hc;
}

static __global__ void k_hscb_bcast_f32(
        const float * residual,
        const float * block,
        const float * inject,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t nt,
        int64_t nb_r0, int64_t nb_r1, int64_t nb_r2,
        int64_t nb_b0, int64_t nb_b1,
        int64_t nb_i0, int64_t nb_i1,
        int64_t nb_d0, int64_t nb_d1, int64_t nb_d2,
        float inv_hc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = n_embd * hc * nt;
    if (i >= n) {
        return;
    }
    const int64_t t = i / (n_embd * hc);
    const int64_t r = i - t * n_embd * hc;
    const int64_t h = r / n_embd;
    const int64_t e = r - h * n_embd;

    const float inj = *((const float *) ((const char *) inject + h * nb_i0 + t * nb_i1));
    const float w   = 2.0f / (1.0f + expf(-inj * inv_hc));
    const float b   = *((const float *) ((const char *) block + e * nb_b0 + t * nb_b1));
    const float res = *((const float *) ((const char *) residual + e * nb_r0 + h * nb_r1 + t * nb_r2));
    *((float *) ((char *) dst + e * nb_d0 + h * nb_d1 + t * nb_d2)) = res + b * w;
}

bool ggml_cuda_hscb_try_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) != GGML_HSCB_TAG) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * residual = nullptr;
    const ggml_tensor * block  = nullptr;
    const ggml_tensor * inject = nullptr;
    int64_t hc = 0;
    if (!ggml_cuda_hscb_walk_bcast(dst, &residual, &block, &inject, &hc)) {
        return false;
    }
    if (residual->type != GGML_TYPE_F32 || block->type != GGML_TYPE_F32 || inject->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd = residual->ne[0];
    const int64_t nt     = residual->ne[2];
    const int64_t n      = n_embd * hc * nt;
    const float inv_hc   = 1.0f / (float) hc;

    const dim3 block_dims(256, 1, 1);
    const dim3 grid_dims((unsigned int) ((n + block_dims.x - 1) / block_dims.x), 1, 1);

    k_hscb_bcast_f32<<<grid_dims, block_dims, 0, ctx.stream()>>>(
        (const float *) residual->data,
        (const float *) block->data,
        (const float *) inject->data,
        (float *) dst->data,
        n_embd, hc, nt,
        residual->nb[0] / sizeof(float), residual->nb[1] / sizeof(float), residual->nb[2] / sizeof(float),
        block->nb[0] / sizeof(float), block->nb[1] / sizeof(float),
        inject->nb[0] / sizeof(float), inject->nb[1] / sizeof(float),
        dst->nb[0] / sizeof(float), dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float),
        inv_hc);
    return true;
}
