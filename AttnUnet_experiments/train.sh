
# log_dir logs/{arch}/{dataset}/{loss}/{opt.dropout} 

# New 10.23 : add loss_with_logit option  
# began began_comb
# todo :  began 0.95_not_drop
# pre-train : lr : 5e-4 weight_decay : 5e-4
# fine-tune : lr : 5e-5 weight_decay : 1e-3

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 5e-4 --batch_size 4  \
    --gpus 1 \
    --optim adam \
    --log_dir logs/2ch \
    --save_folder checkpoint/2ch/attnv2/iccad/ssim_ec_ssim --repeat 1 \
    --epoch 450 \
    --dataset iccad \
    --loss ssim_ec_ssim \
    --arch attnv2 \
    --dropout dropblock \
    --dice_q 0.99 \
    --weight_decay 5e-4 \
    --scheduler cosineanealing \
    --img_size 512 \
    --in_ch 2 \
    --monitor f1 \
    --pdn_zeros True \
    # --pdn_density_drop 0.05 \
    # --finetune True
    # --post_fix 2ch \
    # --mixed_precision True \
    # --post_fix  \
    # --loss_with_logit False \

