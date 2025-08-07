#!/bin/bash
# 指定要检查的进程号
target_pid=1014532

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

target_pid=1014533
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

# bash /home/yansu/paper/al_batch_all/shell/other.sh

cd /home/yansu/paper/al_batch_all || exit


#CUDA_VISIBLE_DEVICES=4 python train.py \
#--seed 4593127 \
#--strategy EntropySampling \
#--exp_name EntropySampling1 \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_graphene_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling1/0.029821 &
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--seed 4593127 \
#--strategy MarginSampling \
#--exp_name MarginSampling1 \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_graphene_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling1/0.029821 &
#
#
#CUDA_VISIBLE_DEVICES=6 python train.py \
#--seed 6937481 \
#--exp_name EntropySampling2 \
#--strategy EntropySampling \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_graphene_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling2/0.029821 &
#
#CUDA_VISIBLE_DEVICES=7 python train.py \
#--seed 6937481 \
#--exp_name MarginSampling2 \
#--strategy MarginSampling \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_graphene_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling2/0.029821 &
#

CUDA_VISIBLE_DEVICES=2 python train.py \
--seed 8452937 \
--exp_name EntropySampling3 \
--strategy EntropySampling \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_other.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling3/0.029821 &

CUDA_VISIBLE_DEVICES=3 python train.py \
--seed 8452937 \
--exp_name MarginSampling3 \
--strategy MarginSampling \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_other.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling3/0.029821 &
