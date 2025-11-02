#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstring>
#include "kernels.cuh"

#define CUDA_CHECK(expr) do { \
  cudaError_t _e = (expr); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

#define CUBLAS_CHECK(expr) do { \
  cublasStatus_t _s = (expr); \
  if (_s != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

struct Args {
  int W=512, H=512, C=3, B=8;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;++i){
    if (!strcmp(argv[i],"--w") && i+1<argc) a.W = std::atoi(argv[++i]);
    else if (!strcmp(argv[i],"--h") && i+1<argc) a.H = std::atoi(argv[++i]);
    else if (!strcmp(argv[i],"--c") && i+1<argc) a.C = std::atoi(argv[++i]);
    else if (!strcmp(argv[i],"--batch") && i+1<argc) a.B = std::atoi(argv[++i]);
  }
  return a;
}

double ms_since(std::chrono::high_resolution_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);
  const int B=a.B, H=a.H, W=a.W, C=a.C;

  std::cout << "Config: B="<<B<<" H="<<H<<" W="<<W<<" C="<<C<<"\n";

  size_t n_in = (size_t)B*H*W*C;
  size_t n_out = (size_t)B*H*W; // single output channel
  size_t n_f = 3*3*C;

  // Allocate host pinned memory
  float *h_in, *h_out, *h_f;
  CUDA_CHECK(cudaMallocHost(&h_in,  n_in  * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_out, n_out * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_f,   n_f   * sizeof(float)));

  // Initialize host buffers
  for (size_t i=0;i<n_in;++i) h_in[i] = (float)((i % 13) - 6) * 0.1f;
  for (size_t i=0;i<n_f;++i)  h_f[i]  = 0.1111f; // simple blur-like kernel

  // Allocate device memory
  float *d_in, *d_relu, *d_out, *d_f;
  CUDA_CHECK(cudaMalloc(&d_in,  n_in*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_relu,n_in*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n_out*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_f,   n_f*sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // H2D
  auto t0 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, n_in*sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_f,  h_f,  n_f *sizeof(float), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "H2D(ms): " << ms_since(t0) << "\n";

  // ReLU
  t0 = std::chrono::high_resolution_clock::now();
  relu_forward(d_in, d_relu, (int)n_in, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "ReLU(ms): " << ms_since(t0) << "\n";

  // Conv3x3 NHWC (naive)
  t0 = std::chrono::high_resolution_clock::now();
  conv3x3_nhwc_naive(d_in, d_f, d_out, B, H, W, C, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "Conv3x3 naive(ms): " << ms_since(t0) << "\n";

  // cuBLAS GEMM baseline: (B*H*W, C) * (C, K) -> (B*H*W, K), with K=1
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  const int M = B*H*W;
  const int N = 1;
  const int K = C;
  float alpha = 1.0f, beta = 0.0f;

  // Allocate Wgemm and Y (we'll reuse d_in as X)
  float *d_Wgemm, *d_Y;
  CUDA_CHECK(cudaMalloc(&d_Wgemm, K*N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Y, M*N*sizeof(float)));
  CUDA_CHECK(cudaMemsetAsync(d_Wgemm, 0, K*N*sizeof(float), stream));

  t0 = std::chrono::high_resolution_clock::now();
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           /*m*/ M, /*n*/ N, /*k*/ K,
                           &alpha,
                           /*A*/ d_in, /*lda*/ M,  // treat NHWC as (M,K) contiguous along K
                           /*B*/ d_Wgemm, /*ldb*/ K,
                           &beta,
                           /*C*/ d_Y, /*ldc*/ M));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "cuBLAS GEMM(ms): " << ms_since(t0) << "\n";

  // D2H
  t0 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, n_out*sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "D2H(ms): " << ms_since(t0) << "\n";

  double sum = 0.0;
  for (size_t i=0;i<n_out;i+= (n_out/16 + 1)) sum += h_out[i];
  std::cout << "Checksum(sampled): " << sum << "\n";

  // Cleanup
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_relu));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_f));
  CUDA_CHECK(cudaFree(d_Wgemm));
  CUDA_CHECK(cudaFree(d_Y));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaFreeHost(h_f));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
