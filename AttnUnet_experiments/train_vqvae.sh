# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=1 python3 train_vqvae.py --lr 5e-4 --batch_size 2 --repeat 1 \
    --gpus 1 \
    --log_dir logs \
    --monitor f1 \
    --dataset cus \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/3ch/attnv6/cus_new/ssim_dice_mse \
    --arch attnv6 \
    --in_ch 3 \
    --loss ssim_dice_mse \
    --dice_q 0.99 \
    --img_size 256 \
    --use_ema False \
    --post_fix 1um/embed64/relu \
    --dbu_per_px 1um \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --use_raw True
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \