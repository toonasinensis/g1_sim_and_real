cmake_minimum_required(VERSION 3.18)
project(trt_infer_example LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# 找到 CUDA
# -----------------------------
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found!")
endif()


set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.4")  # 替换为 CUDA 11.4 的实际路径
find_package(OpenCV REQUIRED)

# 找到CUDA 库


# find_package(CUDA REQUIRED)
set(CUDA_ROOT /usr/local/cuda)
include_directories(${CUDA_ROOT}/include/)
link_directories(${CUDA_ROOT}/lib64)
# tensorRT
# find_package(TensorRT REQUIRED)
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)


# 找到CUDA 库
# find_package(CUDA REQUIRED)
set(CUDA_ROOT /usr/local/cuda)
include_directories(${CUDA_ROOT}/include/)
link_directories(${CUDA_ROOT}/lib64)
# tensorRT
# find_package(TensorRT REQUIRED)
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)


# -----------------------------
# TensorRT 路径（根据你本机安装调整）
# 假设 Orin 上 TensorRT 安装在 /usr/local/TensorRT
# -----------------------------
set(TENSORRT_DIR "/usr/local/TensorRT")  # <-- 修改为你实际路径
include_directories(
    ${TENSORRT_DIR}/include
    ${CUDA_INCLUDE_DIRS}
y
)
link_directories(
    ${TENSORRT_DIR}/lib
)

# -----------------------------
# 可执行文件
# -----------------------------
add_executable(trt_infer
    main.cpp  # 你的 cpp 文件名
)

# -----------------------------
# 链接库
# -----------------------------
target_link_libraries(trt_infer
                    -lpthread 
                      -lm 
                      -lrt 
                      -ldl 
                      -lstdc++
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
                      nvinfer
                      nvonnxparser
                      cuda
                      cudart
                      nvinfer 
                      nvonnxparser 
                      nvparsers 
                      nvinfer_plugin
)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_61,code=sm_61")