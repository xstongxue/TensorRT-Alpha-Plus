#!/bin/bash
set -e  # 遇到错误时退出

# 查找所有非 .sh 文件并删除
find . -type f ! -name "*.sh" -delete

# 重新配置和编译
cmake ..
make -j10