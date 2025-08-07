# bash /home/yansu/paper/batch/shell/MoS2/batch1_0123.sh
cd /home/yansu/paper/batch || exit
mat_name='MoS2'
max_iters=60000
#max_iters=100
i_batch=1
save_dir='runsbatch_MoS2'


#model_name='FlashInternImage'
#cfg_filename='upernet_flash_internimage_b_in1k_768.py'
#crop_size=768
#CUDA_VISIBLE_DEVICES=7 python ./train.py \
#     --config ./config/${cfg_filename} \
#     --config-merge ./config/dataset_MoS2/2024_annlab_${mat_name}_batch${i_batch}_${crop_size}.py \
#     --work-dir "./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/work_dirs" \
#     --cfg-options cfg.visualizer.vis_backends[0].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/local" \
#     --cfg-options cfg.visualizer.vis_backends[1].save_dir="./${save_dir}/${mat_name}_batch${i_batch}_${model_name}/mlruns" \
#     --cfg-options cfg.visualizer.vis_backends[1].exp_name="batch${i_batch}" \
#     --cfg-options cfg.visualizer.vis_backends[1].run_name="${i_batch}/${model_name}" \
#     --cfg-options cfg.param_scheduler[1].end=${max_iters} \
#     --cfg-options cfg.train_cfg.max_iters=${max_iters} \
#     --cfg-options cfg.train_cfg.val_interval=$(($max_iters / 10)) \
#     --cfg-options cfg.default_hooks.checkpoint.interval=$(($max_iters / 10)) &




model_name='BiFormer'
cfg_filename='upernet_biformer_b_in1k_768.py'
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



model_name='UniRepLKNet'
cfg_filename='upernet_unireplknet_b_in22k_768.py'
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


model_name='ViTCoMer'
cfg_filename='upernet_vit_comer_b_in22k_640.py'
crop_size=640
CUDA_VISIBLE_DEVICES=6 python ./train.py \
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