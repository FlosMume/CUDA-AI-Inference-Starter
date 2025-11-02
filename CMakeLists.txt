
cmake_minimum_required(VERSION 3.18)
project(cuda_ai_inference LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Optional: Find CUDA Toolkit (only needed if you want to use toolkit components directly)
find_package(CUDAToolkit REQUIRED)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.cu)

# Create executable
add_executable(cuda_ai_inference ${SOURCES})

# Include directories
target_include_directories(cuda_ai_inference PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries if needed
target_link_libraries(cuda_ai_inference PRIVATE
    ${CUDAToolkit_LIBRARIES}
)

# Set CUDA compilation flags
set_target_properties(cuda_ai_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
