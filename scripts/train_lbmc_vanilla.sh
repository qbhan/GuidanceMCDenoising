python train_lbmc.py \
    --single_gpu \
    --batch_size 8 \
    --val_epoch 1 \
    --data_dir /mnt/ssd2/kbhan/SBMC \
    --model_name LBMC \
    --desc "LBMC" \
    --num_epoch 10 \
    --lr_dncnn 1e-4 \
    --device_id 2