#include "kernels.cuh"
#include <cuda_runtime.h>

// ------------------------------------
// ReLU
// ------------------------------------
__global__ void relu_kernel(const float* __restrict__ x,
                            float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v > 0.f ? v : 0.f;
  }
}

void relu_forward(const float* d_in, float* d_out, int n, cudaStream_t stream) {
  const int tpblock = 256;
  int blocks = (n + tpblock - 1) / tpblock;
  relu_kernel<<<blocks, tpblock, 0, stream>>>(d_in, d_out, n);
}

// ------------------------------------
// Naive 3x3 NHWC convolution
// ------------------------------------
__global__ void conv3x3_nhwc_kernel(const float* __restrict__ in,
                                    const float* __restrict__ filt, // [3,3,C]
                                    float* __restrict__ out,
                                    int B, int H, int W, int C) {
  // Each thread computes one output element: (b, h, w)
  int b = blockIdx.z;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H || w >= W) return;

  // Pad=1, stride=1
  float acc = 0.f;
  for (int kh = -1; kh <= 1; ++kh) {
    int ih = h + kh;
    if (ih < 0 || ih >= H) continue;
    for (int kw = -1; kw <= 1; ++kw) {
      int iw = w + kw;
      if (iw < 0 || iw >= W) continue;
      int fh = kh + 1;
      int fw = kw + 1;
      // sum over channels
      for (int c = 0; c < C; ++c) {
        float v = in[((b * H + ih) * W + iw) * C + c];
        float f = filt[(fh * 3 + fw) * C + c];
        acc += v * f;
      }
    }
  }
  // Write to single-channel output
  out[((b * H + h) * W + w)] = acc;
}

void conv3x3_nhwc_naive(const float* d_in, const float* d_filt,
                        float* d_out, int B, int H, int W, int C,
                        cudaStream_t stream) {
  dim3 block(16, 16, 1);
  dim3 grid((W + block.x - 1) / block.x,
            (H + block.y - 1) / block.y,
            B);
  conv3x3_nhwc_kernel<<<grid, block, 0, stream>>>(d_in, d_filt, d_out, B, H, W, C);
}
