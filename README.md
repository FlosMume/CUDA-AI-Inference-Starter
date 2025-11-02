# CUDA AI Inference Starter (RTX 4070 SUPER)

A compact, production-style starter project for your workstation (Windows 11 + WSL2 Ubuntu 22.04, CUDA 12.x, RTX 4070 SUPER).  
It demonstrates a **deep-learning inference micro-pipeline** with:

- **Custom CUDA kernels** (ReLU and 2D 3x3 convolution, NHWC)
- **cuBLAS GEMM baseline** (for dense layers)
- **Pinned memory + streams** for overlap (host→device, compute, device→host)
- **Profiling hooks** for Nsight Systems & Nsight Compute
- **CMake** build (C++17, CUDA 12.x)

> Goal: A clean scaffold you can extend toward ResNet/UNet-lite, Tensor Cores, and TensorRT.

---

## 1) Requirements

- NVIDIA driver supporting CUDA 12.x
- CUDA Toolkit 12.x (`nvcc --version`)
- CMake ≥ 3.24, g++ ≥ 10
- (Optional) Nsight Systems (`nsys`) / Nsight Compute (`ncu`)

On WSL2 Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake
# CUDA installed via NVIDIA instructions
```

---

## 2) Build & Run

```bash
# Configure & build (Release by default)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run with default sizes
./build/cuda_ai_infer --w 512 --h 512 --c 3 --batch 8
```

You’ll see timings for:
- H2D/D2H copies (pinned vs pageable)
- ReLU kernel
- 3x3 conv kernel (naive, NHWC)
- cuBLAS GEMM (as dense layer baseline)

---

## 3) Profile

### Nsight Systems (timeline)
```bash
./scripts/profile_nsys.sh ./build/cuda_ai_infer --w 1024 --h 1024 --c 3 --batch 8
# Produces cuda_ai_infer.qdrep (open in Nsight Systems GUI)
```

### Nsight Compute (kernel metrics)
```bash
ncu --set full ./build/cuda_ai_infer --w 1024 --h 1024 --c 3 --batch 8
```

---

## 4) Project Layout

```
cuda-ai-inference-starter/
  ├── CMakeLists.txt
  ├── README.md
  ├── src/
  │   ├── main.cu
  │   ├── kernels.cuh
  │   └── kernels.cu
  └── scripts/
      └── profile_nsys.sh
```

---

## 5) Next Steps (suggested extensions)

- Replace naive conv with **shared-memory tiled** conv and measure speedup.
- Switch to **NCHW** and use **Tensor Cores** (TF32/FP16) via cuDNN or CUTLASS.
- Add **TensorRT** path for an ONNX model (e.g., ResNet-50) to compare latency.
- Introduce **multiple CUDA streams** and overlap H2D/D2H with compute.
- Add image I/O (stb_image) to run on real images and verify numerics.

---

## 6) Notes

- The conv kernel here is a readable baseline (not optimized). It’s ideal for learning and then iterating with shared memory, vectorized loads, and better occupancy.
- GEMM uses cuBLAS (`cublasSgemm`) to offer a dense-layer baseline and confirm that your toolkit is wired correctly.
