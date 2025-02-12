# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=1 python3 train_vq_multi_task.py --lr 5e-4 --batch_size 1 --repeat 1 \
    --gpus 1 \
    --log_dir logs \
    --monitor ssim \
    --dataset cus \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/auto_encodoer/sr_v2/cus_new/ms_ssim_mae_ec_ssim \
    --arch sr_v2 \
    --in_ch 1 \
    --loss ms_ssim_mae_ec_ssim \
    --dice_q 0.99 \
    --img_size 256 \
    --num_embeddings 64 \
    --auto_encoder \
    # --post_fix  \
    # --dbu_per_px 1um \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --use_raw True
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \