
# for began-256  size

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 5e-5 --batch_size 1  \
    --gpus 1 \
    --optim adam \
    --log_dir logs/2ch_res/256_b1 \
    --save_folder checkpoint/2ch/attnv2/asap7/ssim_mae_256_b1 --repeat 1 \
    --epoch 600 \
    --dataset asap7 \
    --loss ms_ssim \
    --arch attnv2 \
    --dropout dropblock \
    --dice_q 0.99 \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --img_size 256 \
    --in_ch 2 \
    --monitor f1 \
    --finetune True \
    # --post_fix iccad_real_w_dcay_1e5
    # --pdn_zeros True \
    # --mixed_precision True \
    # --post_fix  \
    # --loss_with_logit False \
    # --pdn_density_drop 0.05 \

