# bash /home/yansu/paper/batch/shell/graphene/batch5_0123.sh






cd /home/yansu/paper/batch || exit
#max_iters_list=(0 40000 40000 60000 70000 80000)


mat_name='MoS2'
max_iters=130000
#max_iters=100
i_batch=5
save_dir='runsbatch5_reversion'


#bash /home/yansu/paper/wait_for_gpu.sh 7

model_name='DeeplabV3Plus'
cfg_filename='deeplabv3plus_r101_in1k_768.py'
crop_size=768
CUDA_VISIBLE_DEVICES=4 python ./train.py \
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
     --cfg-options cfg.default_hooks.checkpoint.interval=$(($max_iters / 10)) &

model_name='Unet'
cfg_filename='unet_768.py'
crop_size=768
CUDA_VISIBLE_DEVICES=5 python ./train.py \
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
     --cfg-options cfg.default_hooks.checkpoint.interval=$(($max_iters / 10)) &