python train_kpcn.py \
    --single_gpu \
    --batch_size 8 \
    --val_epoch 1 \
    --data_dir /mnt/ssd1/iycho/KPCN \
    --model_name KPCN_full \
    --desc "KPCN_full" \
    --num_epoch 10 \
    --lr_dncnn 1e-4 \
    --train_branches \
    --device_id 3
