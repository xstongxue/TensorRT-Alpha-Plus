# 使用国内 NVIDIA 镜像源
FROM nvcr.io.cn-beijing.volces.com/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu18.04

# 使用清华源加速
RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list && \
    sed -i 's#http://security.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list && \
    apt-get update

# 安装基础开发工具和 Python
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gdb \
    cmake \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 \
    && update-alternatives --config python3

# 创建并设置 TensorRT 目录
RUN mkdir -p /data02/xs/app/
COPY TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.7.tar.gz /data02/xs/app/
RUN cd /data02/xs/app/ && \
    tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.7.tar.gz && \
    rm TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.7.tar.gz && \
    mkdir -p workspace

# 设置 TensorRT 环境变量
ENV TensorRT_ROOT=/data02/xs/app/TensorRT-8.6.1.6
ENV LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:${LD_LIBRARY_PATH}

# 安装 OpenCV 依赖
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libopencv-dev \
    && apt-get clean

# 设置工作目录
WORKDIR /data02/xs/app/workspace

# 设置用户
RUN useradd -m -s /bin/bash xstongxue
USER xstongxue