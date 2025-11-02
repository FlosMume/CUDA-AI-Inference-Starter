
cmake_minimum_required(VERSION 3.18)
project(cuda_ai_inference LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA Toolkit (modern CUDAToolkit package)
find_package(CUDAToolkit REQUIRED)

# Source files (explicit is clearer than glob for small projects)
set(SOURCES
  src/main.cu
  src/kernels.cu
)

# Create executable target
add_executable(cuda_ai_inference ${SOURCES})

# Include directories
target_include_directories(cuda_ai_inference PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA runtime + cuBLAS
target_link_libraries(cuda_ai_inference PRIVATE
  CUDA::cudart
  CUDA::cublas
)

# Useful CUDA properties
set_target_properties(cuda_ai_inference PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  # RTX 4070 SUPER (Ada) = SM 8.9
  CUDA_ARCHITECTURES 89
)


# Add line info for Nsight profiling even in Release
target_compile_options(cuda_ai_inference PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
