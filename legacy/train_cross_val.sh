
# log_dir logs/{arch}/{dataset}/{loss}/{opt.dropout} 

# curr :  iccad ms_ssim_cosanealing, ms_ssim_mae_cosanealing pt
# todo :  iccad ms_ssim_cosanealing, ms_ssim_mae_cosanealing ft
# pre-train : lr : 5e-4 weight_decay : 5e-4
# fine-tune : lr : 5e-5 weight_decay : 1e-3

CUDA_VISIBLE_DEVICES=3 python3 train_cross_val.py --lr 5e-5 --batch_size 1  \
    --gpus 1 \
    --optim adam \
    --log_dir logs \
    --save_folder checkpoint/2ch/attnv2/asap7/ssim_mae_256_b1 --repeat 1 \
    --epoch 600 \
    --loss ms_ssim \
    --arch attnv2 \
    --dropout dropblock \
    --dice_q 0.99 \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --finetune True \
    --img_size 256 \
    --in_ch 2 \
    # --post_fix zeros \
    # --loss_with_logit False \
    # --pdn_density_drop 0.05 \

