

sleep 2h
bash /home/yansu/paper/wait_for_gpu.sh 5

cd /home/yansu/paper/al_batch_all || exit


#
#CUDA_VISIBLE_DEVICES=1 python train.py \
#--seed 4593127 \
#--strategy RandomSampling \
#--exp_name MarginSampling \
#--work_dir ./run_important \
#--cfg_merge ./config/s3_e30_n10_important.py \
#            ./config/2024_annlab_graphene_batch5_768.py \
#--cfg ./config/upernet_flash_internimage_b_in1k_768.py


CUDA_VISIBLE_DEVICES=5 python train.py \
--seed 4593127 \
--strategy RandomSampling \
--exp_name MarginSampling38 \
--work_dir ./run_important \
--cfg_merge ./config/s3_e30_n10_important_38.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py


CUDA_VISIBLE_DEVICES=5 python train.py \
--seed 4593127 \
--strategy RandomSampling \
--exp_name MarginSampling78 \
--work_dir ./run_important \
--cfg_merge ./config/s3_e30_n10_important_78.py \
            ./config/2024_annlab_graphene_batch5_768.py \
--cfg ./config/upernet_flash_internimage_b_in1k_768.py