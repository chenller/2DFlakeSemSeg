#!/bin/bash
# 指定要检查的进程号
#target_pid=3039886
#
## 循环检查直到目标进程不在yansu相关的进程中
#while true; do
#    # 获取名为yansu的所有进程的PID列表
#    yansu_pids=$(ps -ef | grep 'yansu' | grep -v grep | awk '{print $2}')
#
#    # 检查目标PID是否在yansu_pids列表中
#    if echo "$yansu_pids" | grep -q "$target_pid"; then
#        echo "Process $target_pid associated with yansu is running, waiting for 1 minute..."
#        sleep 60  # 等待60秒
#    else
#        echo "Process $target_pid is no longer associated with yansu, exiting loop."
#        break  # 目标进程不在yansu相关的进程中，退出循环
#    fi
#done

# bash /home/yansu/paper/batch/shell/mos2/batch_sample.sh
#bash /home/yansu/paper/wait_for_gpu.sh 7

#cd /home/yansu/paper/batch/sample || exit
#CUDA_VISIBLE_DEVICES=7 python /home/yansu/paper/batch/sample/sample2.py


cd /home/yansu/paper/batch || exit
mat_name='MoS2'
max_iters=70000
#max_iters=100
i_batch=2
save_dir='runsbatch_MoS2'


model_name='FlashInternImage'
cfg_filename='upernet_flash_internimage_b_in1k_768.py'
crop_size=768
CUDA_VISIBLE_DEVICES=2 python ./train.py \
     --config ./config/${cfg_filename} \
     --config-merge ./config/dataset_MoS2/2024_annlab_${mat_name}_batch${i_batch}_${crop_size}.py \
     --work-dir "./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/work_dirs" \
     --cfg-options cfg.visualizer.vis_backends[0].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/local" \
     --cfg-options cfg.visualizer.vis_backends[1].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/mlruns" \
     --cfg-options cfg.visualizer.vis_backends[1].exp_name="batch${i_batch}" \
     --cfg-options cfg.visualizer.vis_backends[1].run_name="${i_batch}/${model_name}" \
     --cfg-options cfg.param_scheduler[1].end=${max_iters} \
     --cfg-options cfg.train_cfg.max_iters=${max_iters} \
     --cfg-options cfg.train_cfg.val_interval=$(($max_iters / 10)) \
     --cfg-options cfg.default_hooks.checkpoint.interval=$(($max_iters / 10))

cd /home/yansu/paper/batch/sample || exit
CUDA_VISIBLE_DEVICES=7 python /home/yansu/paper/batch/sample/sample3.py


cd /home/yansu/paper/batch || exit
mat_name='MoS2'
max_iters=90000
#max_iters=100
i_batch=3
save_dir='runsbatch_MoS2'


model_name='FlashInternImage'
cfg_filename='upernet_flash_internimage_b_in1k_768.py'
crop_size=768
CUDA_VISIBLE_DEVICES=7 python ./train.py \
     --config ./config/${cfg_filename} \
     --config-merge ./config/dataset_MoS2/2024_annlab_${mat_name}_batch${i_batch}_${crop_size}.py \
     --work-dir "./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/work_dirs" \
     --cfg-options cfg.visualizer.vis_backends[0].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/local" \
     --cfg-options cfg.visualizer.vis_backends[1].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/mlruns" \
     --cfg-options cfg.visualizer.vis_backends[1].exp_name="batch${i_batch}" \
     --cfg-options cfg.visualizer.vis_backends[1].run_name="${i_batch}/${model_name}" \
     --cfg-options cfg.param_scheduler[1].end=${max_iters} \
     --cfg-options cfg.train_cfg.max_iters=${max_iters} \
     --cfg-options cfg.train_cfg.val_interval=$(($max_iters / 10)) \
     --cfg-options cfg.default_hooks.checkpoint.interval=$(($max_iters / 10))

cd /home/yansu/paper/batch/sample || exit
CUDA_VISIBLE_DEVICES=7 python /home/yansu/paper/batch/sample/sample4.py