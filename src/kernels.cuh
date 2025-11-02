#pragma once
#include <cuda_runtime.h>

// Simple ReLU kernel: y = max(x, 0)
void relu_forward(const float* d_in, float* d_out, int n, cudaStream_t stream);

// Naive 3x3 NHWC convolution with single output channel (demo baseline).
// in:  [B, H, W, C]
// filt:[3, 3, C]
// out: [B, H, W, 1]
void conv3x3_nhwc_naive(const float* d_in, const float* d_filt,
                        float* d_out, int B, int H, int W, int C,
                        cudaStream_t stream);
