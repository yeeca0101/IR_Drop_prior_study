# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=0 python3 train_vqvae.py --lr 5e-4 --batch_size 2 --repeat 1 \
    --gpus 1 \
    --log_dir logs/6th \
    --monitor f1 \
    --dataset cus \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/6th/3ch/attnv6_1/cus_new/ssim_huber \
    --arch attnv6_1 \
    --in_ch 3 \
    --loss ssim_huber \
    --dice_q 0.99 \
    --img_size 256 \
    --post_fix 1um/top_p_8 \
    --num_embeddings 512 \
    --dbu_per_px 1um \
    --metric_type max
    # --use_raw True \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \