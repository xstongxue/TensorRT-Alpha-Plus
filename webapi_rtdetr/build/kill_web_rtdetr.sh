#!/bin/bash

# 检查并杀死已有的 app_rtdetr 进程
app_name="./web_rtdetr"  # 指定目标应用名称
pic_ss=$(pgrep -f "${app_name}")  # 匹配目标进程名称

if [[ -n $pic_ss ]]; then
    echo "Killing existing ${app_name} process with PID: ${pic_ss}"
    kill ${pic_ss}  # 优雅终止
    sleep 2
    # 确认进程是否结束
    if pgrep -f "${app_name}" > /dev/null; then
        echo "Force killing remaining process..."
        kill -9 ${pic_ss}
    fi
else
    echo "No existing ${app_name} process found."
fi
