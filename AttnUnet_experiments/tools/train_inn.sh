

CUDA_VISIBLE_DEVICES=3 python3 train_inn.py \
    --lr 5e-4 \
    --monitor mae \
    --batch_size 1 \
    --epoch 100 \
    --gpus 1 \
    --dataset cus \
    --in_channels 2 \
    --hidden_channels 64 \
    --num_layers 4 \
    --log_dir logs_inn \
    --save_folder ./checkpoint/inn \
    --dbu_per_px 200nm \
    --mixed_precision False
