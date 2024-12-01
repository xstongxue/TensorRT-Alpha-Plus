#!/bin/bash
set -e  # 遇到错误时退出

# 重新配置和编译
make clean
make -j10