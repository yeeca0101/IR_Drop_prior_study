# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=0 python3 train_vqvae.py --lr 5e-4 --batch_size 16 --repeat 1 \
    --gpus 1 \
    --optim adamw \
    --log_dir logs \
    --save_folder checkpoint/2ch/attnv5_2/cus_new/ssim_dice_mse \
    --epoch 750 \
    --dataset cus \
    --loss ssim_dice_mse \
    --arch attnv5_2 \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --img_size 256 \
    --in_ch 2 \
    --monitor f1 \
    --use_ema False \
    --post_fix 1um/sq_ema_false_b16 \
    --dbu_per_px 1um \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --use_raw True
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \