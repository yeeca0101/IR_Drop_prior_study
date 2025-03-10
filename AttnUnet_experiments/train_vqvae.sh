# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=0 python3 train_vqvae.py --lr 5e-4 --batch_size 4 --repeat 1 \
    --gpus 1 \
    --log_dir logs \
    --monitor f1 \
    --dataset cus \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/test/3ch/attnv6_1/cus_new/ssim_mae \
    --arch attnv6_1 \
    --in_ch 3 \
    --loss ssim_mae \
    --dice_q 0.99 \
    --img_size 256 \
    --post_fix 200nm/b4 \
    --num_embeddings 64 \
    --dbu_per_px 200nm \
    # --use_raw True \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \