#!/bin/bash

# 检查并杀死已有的 app_rtdetr 进程
app_name="./web_rtdetr"  # 指定目标应用名称
pic_ss=$(pgrep -f "${app_name}")  # 匹配目标进程名称，确保不匹配脚本自身

if [[ -n $pic_ss ]]; then
    echo "Killing existing ${app_name} process with PID: ${pic_ss}"
    kill ${pic_ss}  # 优雅终止
    sleep 2
    # 确认进程是否结束
    if pgrep -f "${app_name}$" > /dev/null; then
        echo "Force killing remaining process..."
        kill -9 ${pic_ss}
    fi
else
    echo "No existing ${app_name} process found."
fi

# 环境激活（如果需要）
# source /opt/conda/etc/profile.d/conda.sh
# conda activate base

# 参数设置
model_path="../../weights/rtdetr_dla.trt"
device=0  # 指定设备
port=8888 # 端口号

# 参数检查
if [[ ! -f ${model_path} ]]; then
    echo "Error: Model file not found at ${model_path}."
    exit 1
fi

# 运行应用
log_file="rtdetr_detect_result.log"
echo "Starting ${app_name} with model: ${model_path}, device: ${device}, port: ${port}"
CUDA_VISIBLE_DEVICES=1 nohup ${app_name} \
    ${model_path} \
    ${device} \
    ${port} \
> ${log_file} 2>&1 &

# 显示启动信息
echo "${app_name} started. Logs: ${log_file}"
