#!/bin/bash
# 指定要检查的进程号
target_pid=4095759

# 循环检查直到目标进程不在yansu相关的进程中
while true; do
    # 获取名为yansu的所有进程的PID列表
    yansu_pids=$(ps -ef | grep 'yansu' | grep -v grep | awk '{print $2}')

    # 检查目标PID是否在yansu_pids列表中
    if echo "$yansu_pids" | grep -q "$target_pid"; then
        echo "Process $target_pid associated with yansu is running, waiting for 1 minute..."
        sleep 60  # 等待60秒
    else
        echo "Process $target_pid is no longer associated with yansu, exiting loop."
        break  # 目标进程不在yansu相关的进程中，退出循环
    fi
done
# 执行其他命令
echo "Executing other commands..."
# 在这里添加你想执行的其他命令

sleep 30  # 等待60秒

cd /home/yansu/mmlabmat/paper/batchtrain/codeal
CUDA_VISIBLE_DEVICES=3 python /home/yansu/mmlabmat/paper/batchtrain/codeal/sample.py
bash /home/yansu/mmlabmat/paper/batchtrain/code/shell/graphene/batch3_023.sh