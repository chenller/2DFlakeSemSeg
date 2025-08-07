
#bash /home/yansu/paper/wait_for_gpu.sh 6

cd /home/yansu/paper/al_batch_all || exit

#source /home/yansu/.bashrc
#mmlab

#CUDA_VISIBLE_DEVICES=3 python train.py \
#--seed 1843297 \
#--strategy RandomSampling \
#--exp_name RandomSampling1 \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_random.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py &




CUDA_VISIBLE_DEVICES=0 python train.py \
--seed 893262 \
--strategy RandomSampling \
--exp_name RandomSampling2 \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_random.py \
            ./config/2024_annlab_MoS2_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py &

CUDA_VISIBLE_DEVICES=3 python train.py \
--seed 9746258 \
--strategy RandomSampling \
--exp_name RandomSampling3 \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_random.py \
            ./config/2024_annlab_MoS2_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py &



CUDA_VISIBLE_DEVICES=4 python train.py \
--seed 4581247 \
--strategy RandomSampling \
--exp_name RandomSampling4 \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_random.py \
            ./config/2024_annlab_MoS2_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py &




#CUDA_VISIBLE_DEVICES=3 python train.py \
#--seed 8617534 \
#--strategy RandomSampling \
#--exp_name RandomSampling5 \
#--work_dir ./run_s3_e30_n10 \
#--cfg_merge ./config/s3_e30_n10_random.py \
#            ./config/2024_annlab_MoS2_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py &