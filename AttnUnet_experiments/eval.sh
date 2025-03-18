

CUDA_VISIBLE_DEVICES=2 python3 evaluate.py \
    --img_size 512 \
    --batch_size 1 \
    --in_ch 12 \
    --dataset hidden \
    --arch attnv6_1 \
    --num_embeddings 512 \
    --pdn_zeros \
    --post_min_max \
    --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/ssim_huber/embed512_mtr_max/finetune/iccad/ms_ssim_dice/max_norm_mtr_max_8186_pdn_zeros
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/ssim_huber/embed512_mtr_max/finetune/iccad/ssim_huber/max_norm_mtr_max_7978_pdn_zeros
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/ssim_huber/embed512_mtr_max/finetune/iccad/ssim_huber/max_norm_mtr_max_7881



    # --post_min_max \
    
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/began/ssim_ec_ssim/default/100nm/finetune/iccad/ms_ssim_dice/100_norm_b1_8vs2_target_mm_ssim

    


    # 