#pragma once

#include "common.cuh"

// Tagged HC scatter-combine (Qwen4exp HSCB). Returns true if dispatched.
bool ggml_cuda_hscb_try_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
