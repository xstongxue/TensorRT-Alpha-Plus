cmake_minimum_required(VERSION 3.10)
PROJECT(rtdetr_web VERSION 1.0.0 LANGUAGES C CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)  # 使用 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)  # 可设置为 Release
#set(CMAKE_BUILD_TYPE Release)

# 设置输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# 设置 CMake RPATH 路径
set(CMAKE_SKIP_BUILD_RPATH False)
set(CMAKE_SKIP_RPATH False)
set(CMAKE_BUILD_RPATH "/data02/xs/app/TensorRT-8.6.1.6/lib")

# 查找源码文件
file(GLOB_RECURSE CPPS
    ${PROJECT_SOURCE_DIR}/apps/*.cpp
    ${PROJECT_SOURCE_DIR}/apps/*.cu
    ${PROJECT_SOURCE_DIR}/trt_common/*.cpp
    ${PROJECT_SOURCE_DIR}/trt_common/*.cu
)

# 设置 CUDA 和 TensorRT 的路径
set(CUDA_DIR "/usr/local/cuda")
set(TENSORRT_DIR "/data02/xs/app/TensorRT-8.6.1.6")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 添加头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${TENSORRT_DIR}/include
    "/usr/include/eigen3"
)

# 添加库文件路径
link_directories(
    ${CUDA_DIR}/lib64
    ${TENSORRT_DIR}/lib
)

# 设置 Muduo 库路径，确保能找到 Muduo 的静态库
set(MUDUO_DIR "/usr/local/muduo")
include_directories("${MUDUO_DIR}/include")
link_directories("${MUDUO_DIR}/lib")

# 添加需要链接的库
list(APPEND ALL_LIBS
    cuda cublas cudart cudnn
    nvinfer nvinfer_plugin nvonnxparser
    pthread ${OpenCV_LIBS}
    muduo_net muduo_base  # 添加 Muduo 库
)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wfatal-errors -pthread -w")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -g -O0 -Xcompiler -fPIC")

# 创建共享库
add_library(${PROJECT_NAME} SHARED ${CPPS})
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})

# 设置 CUDA 架构
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 86)

# 编译选项
target_compile_options(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread -lineinfo --use_fast_math --disable-warnings>
)

# 创建可执行文件并链接 Muduo 和其他依赖库
add_executable(web_rtdetr main.cpp)  
target_link_libraries(web_rtdetr ${PROJECT_NAME} ${ALL_LIBS})