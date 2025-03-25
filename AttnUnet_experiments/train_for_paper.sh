# TODO: include VQVAE

# 1e3, 1e3
# fine 5e4, 1e3

CUDA_VISIBLE_DEVICES=2 python3 train_vqvae.py --lr 5e-5 --batch_size 2 --repeat 1 \
    --gpus 1 \
    --log_dir logs \
    --monitor f1 \
    --dataset iccad_real_hidden \
    --optim adamw \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --epoch 750 \
    --save_folder checkpoint/paper \
    --arch attnv7 \
    --in_ch 12 \
    --loss ms_ssim \
    --dice_q 0.92 \
    --img_size 512 \
    --num_embeddings 512 \
    --dbu_per_px 1um \
    --metric_type max \
    --post_fix g_max \
    --target_norm g_max \
    --inp_norm g_max \
    --top_percent 0.8 \
    --finetune \
    --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/paper/12/attnv7/iccad/ssim_huber/1um/min_max/
    # --use_raw True \
    # --post_min_max True \