cmake_minimum_required(VERSION 3.24)
project(surgscene_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include(FetchContent)
FetchContent_Declare(eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG 3.4.0 GIT_SHALLOW TRUE)
FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2 GIT_SHALLOW TRUE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_Populate(eigen)  # header-only; skip its own CMake project
FetchContent_MakeAvailable(googletest)

# CUDA runtime + TensorRT come from the Python environment's wheels (no system CUDA toolkit needed).
set(VENV_SITE "${CMAKE_SOURCE_DIR}/../.venv/lib/python3.12/site-packages" CACHE PATH "uv site-packages")
set(CUDART_DIR "${VENV_SITE}/nvidia/cuda_runtime")
set(TRT_LIB_DIR "${VENV_SITE}/tensorrt_libs")
set(TRT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorrt_oss/include")
foreach(p "${CUDART_DIR}/include/cuda_runtime_api.h" "${TRT_LIB_DIR}/libnvinfer.so.10" "${TRT_INCLUDE_DIR}/NvInfer.h")
  if(NOT EXISTS "${p}")
    message(FATAL_ERROR "missing ${p}: run `uv sync --group deploy` and cpp/scripts/fetch_deps.sh")
  endif()
endforeach()

add_library(cudart SHARED IMPORTED)
# crt/host_defines.h lives in the nvcc wheel (headers only; nothing here is compiled with nvcc)
set_target_properties(cudart PROPERTIES IMPORTED_LOCATION "${CUDART_DIR}/lib/libcudart.so.12"
                      INTERFACE_INCLUDE_DIRECTORIES "${CUDART_DIR}/include;${VENV_SITE}/nvidia/cuda_nvcc/include")
add_library(nvinfer SHARED IMPORTED)
set_target_properties(nvinfer PROPERTIES IMPORTED_LOCATION "${TRT_LIB_DIR}/libnvinfer.so.10"
                      INTERFACE_INCLUDE_DIRECTORIES "${TRT_INCLUDE_DIR}")

# Pure-CPU geometry/inference core: no CUDA dependency, testable anywhere
add_library(surgscene_core src/npy.cpp src/map_solver.cpp src/kalman.cpp src/stereo.cpp)
target_include_directories(surgscene_core PUBLIC include ${eigen_SOURCE_DIR})
target_compile_options(surgscene_core PRIVATE -Wall -Wextra)

add_library(surgscene_rt src/trt_engine.cpp src/pipeline.cpp)
target_link_libraries(surgscene_rt PUBLIC surgscene_core nvinfer cudart)
target_compile_options(surgscene_rt PRIVATE -Wall -Wextra)
set(RT_RPATH "${CUDART_DIR}/lib;${TRT_LIB_DIR};${VENV_SITE}/nvidia/cublas/lib;${VENV_SITE}/nvidia/cudnn/lib")

add_executable(bench src/bench_main.cpp)
target_link_libraries(bench PRIVATE surgscene_rt)
set_target_properties(bench PROPERTIES BUILD_RPATH "${RT_RPATH}")

enable_testing()
add_executable(tests tests/test_core.cpp tests/test_engine.cpp)
target_link_libraries(tests PRIVATE surgscene_rt GTest::gtest_main)
target_compile_definitions(tests PRIVATE GOLDEN_DIR="${CMAKE_SOURCE_DIR}/../runs/deploy/goldens"
                           ENGINE_DIR="${CMAKE_SOURCE_DIR}/../runs/deploy")
set_target_properties(tests PROPERTIES BUILD_RPATH "${RT_RPATH}")
include(GoogleTest)
gtest_discover_tests(tests DISCOVERY_TIMEOUT 60)
