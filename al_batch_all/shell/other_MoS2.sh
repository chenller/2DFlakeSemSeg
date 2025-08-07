#!/bin/bash

#bash /home/yansu/paper/wait_for_gpu.sh 4 5

# bash /home/yansu/paper/al_batch_all/shell/other.sh

cd /home/yansu/paper/al_batch_all || exit


#CUDA_VISIBLE_DEVICES=7 python train.py \
#--seed 1843297 \
#--strategy EntropySampling \
#--exp_name EntropySampling1 \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling1/0.029895 &
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--seed 1843297 \
#--strategy MarginSampling \
#--exp_name MarginSampling1 \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling1/0.029895 &
#
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#--seed 8617534 \
#--exp_name EntropySampling5 \
#--strategy EntropySampling \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling5/0.029895 &
#
#CUDA_VISIBLE_DEVICES=7 python train.py \
#--seed 8617534 \
#--exp_name MarginSampling5 \
#--strategy MarginSampling \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling5/0.029895 &
#



#python /home/yansu/paper/wait_file_exit.py /home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling3/0.029895/work_dir/alstate_0.pkl

#CUDA_VISIBLE_DEVICES=7 python train.py \
#--seed 9746258 \
#--exp_name EntropySampling3 \
#--strategy EntropySampling \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_other.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
#--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10/RandomSampling3/0.029895 &

CUDA_VISIBLE_DEVICES=3 python train.py \
--seed 9746258 \
--exp_name MarginSampling3 \
--strategy MarginSampling \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_other.py \
            ./config/2024_annlab_MoS2_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py \
--cfg-options resume_config.find_from=/home/yansu/paper/al_batch_all/run_s3_e30_n10_MoS2/RandomSampling3/0.029895 &