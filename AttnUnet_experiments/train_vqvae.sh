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
    --save_folder checkpoint/new_arch \
    --arch efficientformers0 \
    --in_ch 25 \
    --loss ssim_huber \
    --dice_q 0.92 \
    --img_size 256 \
    --num_embeddings 512 \
    --dbu_per_px 1um \
    --metric_type max \
    --post_fix 1umnm/min_max \
    --target_norm min_max \
    --inp_norm min_max \
    # --finetune \
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/new_arch/25/attnv7/cus/ssim_huber/1um/min_max
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/new_arch/25/attnv7/cus/ssim_dice_mse/1um/min_max
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/new_arch/25/resnextunet/cus/ssim_huber/1um/min_max
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/new_arch/3/attnv7/cus/ssim_huber/1um/min_max
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/new_arch/3/resnextunet/cus/ssim_huber/1um/min_max
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/eff/25/efficientformers1/cus/ssim_huber/1um
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/25/attnv6_1/cus/ssim_huber/1um
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/3/attnv6_1/cus/ssim_huber/1um/all_min_max_norm
    # --checkpoint_path /IR_Drop_prior_study/AttnUnet_experiments/checkpoint/6th/3/attnv6_1/cus/ssim_huber/1um
    # --use_raw True \
    # --post_min_max True \