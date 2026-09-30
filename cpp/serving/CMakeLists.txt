# Holoscan arm of the serving benchmark; built inside the Holoscan container (see run_serving_bench.sh).
cmake_minimum_required(VERSION 3.20)
project(surgscene_serving_holoscan CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(holoscan REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")
find_package(CUDAToolkit REQUIRED)

add_executable(bench_holoscan bench_holoscan.cpp)
target_link_libraries(bench_holoscan PRIVATE holoscan::core holoscan::ops::inference CUDA::cudart)

# the in-process arm, built against the same TensorRT as the Holoscan runtime
add_executable(bench_inproc bench_inproc.cpp ../src/trt_engine.cpp)
target_include_directories(bench_inproc PRIVATE ../include)
target_link_libraries(bench_inproc PRIVATE nvinfer CUDA::cudart)
