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
    --save_folder checkpoint/6th \
    --arch attnv6_1 \
    --in_ch 3 \
    --loss ssim_huber \
    --dice_q 0.995 \
    --img_size 256 \
    --num_embeddings 512 \
    --dbu_per_px 100nm \
    --metric_type max \
    --post_fix 100nm \
    # --finetune \
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/3/attnv6_1/cus/ssim_huber/1um
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/3/attnv6_1/cus/ssim_huber/1um/all_min_max_norm
    # --use_raw True \
    # --post_min_max True \