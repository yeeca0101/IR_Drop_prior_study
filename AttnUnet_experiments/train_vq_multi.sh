# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=1 python3 train_vq_multi_task.py --lr 5e-4 --batch_size 4 --repeat 1 \
    --gpus 1 \
    --log_dir logs \
    --monitor f1 \
    --dataset cus \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/6th \
    --arch cfirst \
    --in_ch 3 \
    --loss huber \
    --loss_aux dice \
    --dice_q 0.99 \
    --img_size 256 \
    --num_embeddings 64 \
    --dbu_per_px 1um \
    --metric_type quantile
    # --post_fix  \
    # --checkpoint_path checkpoint/2ch/attnv5/asap7/ssim_mae/default/adamw/b1
    # --use_raw True
    # --post_min_max True \
    # --finetune True \
    # --vqvae_size large \