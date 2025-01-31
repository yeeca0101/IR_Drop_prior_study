# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=3 python3 train_vqvae.py --lr 5e-4 --batch_size 2 --repeat 1 \
    --gpus 1 \
    --log_dir logs \
    --monitor f1 \
    --dataset cus \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/3ch/attnv6_1/cus_new/ms_ssim_mae_ec_ssim_pice \
    --arch attnv6_1 \
    --in_ch 3 \
    --loss ms_ssim_mae_ec_ssim_pice \
    --dice_q 0.99 \
    --img_size 256 \
    --post_fix 1um/embed512/swisht \
    --num_embeddings 512 \
    --dbu_per_px 1um \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --use_raw True
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \