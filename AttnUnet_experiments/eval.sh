

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
    --img_size 256 \
    --batch_size 1 \
    --in_ch 25 \
    --dataset cus \
    --arch attnv6_1 \
    --num_embeddings 512 \
    --dbu_per_px 1um \
    --th 0.9 \
    --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/25/attnv6_1/cus/ssim_huber/1um/finetune/ssim_huber/1um
    
    
    
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/ssim_huber/embed512_mtr_max/finetune/iccad/ssim_huber/max_norm_mtr_max_7978_pdn_zeros
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/ssim_huber/embed512_mtr_max/finetune/iccad/ssim_huber/max_norm_mtr_max_7881



    # --post_min_max \
    
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/began/ssim_ec_ssim/default/100nm/finetune/iccad/ms_ssim_dice/100_norm_b1_8vs2_target_mm_ssim

    


    # 