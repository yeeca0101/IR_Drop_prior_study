
# log_dir logs/{arch}/{dataset}/{loss}/{opt.dropout} 

# curr :  iccad ms_ssim_cosanealing, ms_ssim_mae_cosanealing pt
# todo :  iccad ms_ssim_cosanealing, ms_ssim_mae_cosanealing ft
# pre-train : lr : 5e-4 weight_decay : 5e-4
# fine-tune : lr : 5e-5 weight_decay : 1e-3

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 5e-5 --batch_size 4  \
    --gpus 1 \
    --optim adam \
    --log_dir logs \
    --save_folder checkpoint/attnv2/iccad/ms_ssim_mae_cosineanealing --repeat 1 \
    --epoch 600 \
    --loss ms_ssim \
    --arch attnv2 \
    --dropout dropblock \
    --dice_q 0.99 \
    --weight_decay 1e-3 \
    --scheduler cosineanealing \
    --finetune True 
    # --post_fix cosineanealing \
    # --loss_with_logit False \
    # --pdn_density_drop 0.05 \

