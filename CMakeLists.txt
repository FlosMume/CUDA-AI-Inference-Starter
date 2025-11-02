cmake_minimum_required(VERSION 3.24)
project(cuda_ai_inference_starter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cuda_ai_infer
  src/main.cu
  src/kernels.cu
)

target_include_directories(cuda_ai_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cuda_ai_infer PRIVATE CUDA::cudart CUDA::cublas)

# Generate debug symbols even in Release for profiling
target_compile_options(cuda_ai_infer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
