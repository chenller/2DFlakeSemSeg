mat_name='graphene'
max_iters=180000
#max_iters=100
i_batch=5
save_dir='runs'

model_name='FlashInternImage'
cfg_filename='upernet_flash_internimage_b_in1k_768.py'
crop_size=768
CUDA_VISIBLE_DEVICES=0 python ./train.py \
     --config ./config/${cfg_filename} \
     --config-merge ./config/dataset/2024_annlab_${mat_name}_batch${i_batch}_${crop_size}.py \
     --work-dir "./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/work_dirs" \
     --cfg-options cfg.visualizer.vis_backends[0].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/local" \
     --cfg-options cfg.visualizer.vis_backends[1].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/mlruns" \
     --cfg-options cfg.visualizer.vis_backends[1].exp_name="batch${i_batch}" \
     --cfg-options cfg.visualizer.vis_backends[1].run_name="${i_batch}/${model_name}" \
     --cfg-options cfg.param_scheduler[1].end=${max_iters} \
     --cfg-options cfg.train_cfg.max_iters=${max_iters} \
     --cfg-options cfg.train_cfg.val_interval=$(($max_iters / 10)) \
     --cfg-options cfg.default_hooks.checkpoint.interval=$(($max_iters / 10))



mat_name='MoS2'
max_iters=130000
#max_iters=100
i_batch=5
save_dir='runs_MoS2'


model_name='FlashInternImage'
cfg_filename='upernet_flash_internimage_b_in1k_768.py'
crop_size=768

CUDA_VISIBLE_DEVICES=0 python ./train.py \
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