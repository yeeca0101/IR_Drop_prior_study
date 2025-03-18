
# 1e3, 1e3
# pre 5e4, 1e3 300
# fine 5e5, 1e3 600

CUDA_VISIBLE_DEVICES=1 python3 train_vqvae.py --lr 5e-5 --batch_size 2 --repeat 1 \
    --gpus 1 \
    --log_dir logs/fine_lab \
    --monitor f1 \
    --dataset iccad \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/paper/12ch/attnv6_1/ms_ssim_dice \
    --arch attnv6_1 \
    --in_ch 12 \
    --loss ms_ssim_dice \
    --dice_q 0.995 \
    --img_size 512 \
    --num_embeddings 512 \
    --post_fix max_norm_mtr_max_8186_pdn_zeros_b2 \
    --metric_type max \
    --finetune  \
    --pdn_zeros  \
    --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/ssim_huber/embed512_mtr_max
    # --dbu_per_px 100nm \
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/began/ssim_ec_ssim/default/100nm
    # --checkpoint_path /workspace/AttnUnet_experiments/checkpoint/paper/12ch/attnv6_1/began/ssim_ec_ssim/default/100nm
    # --mixed_precision True
    # --use_raw True \
    # --post_min_max True \
    # --vqvae_size large \