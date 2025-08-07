cd /home/yansu/paper/al_batch_all || exit



CUDA_VISIBLE_DEVICES=1 python train.py \
--seed 4593127 \
--strategy RandomSampling \
--exp_name RandomSampling1 \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_random.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py &


CUDA_VISIBLE_DEVICES=2 python train.py \
--seed 6937481 \
--strategy RandomSampling \
--exp_name RandomSampling2 \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_random.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py &

CUDA_VISIBLE_DEVICES=3 python train.py \
--seed 8452937 \
--strategy RandomSampling \
--exp_name RandomSampling3 \
--work_dir ./run_s3_e30_n10 \
--cfg_merge ./config/s3_e30_n10_random.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py &