python train_adv.py \
    --single_gpu \
    --batch_size 8 \
    --val_epoch 1 \
    --data_dir /home/kyubeom/ssd1/KPCN \
    --model_name ADV_G_4 \
    --desc "ADV_G_4" \
    --num_epoch 10 \
    --lr_dncnn 1e-4 \
    --device_id 1 \
    --save "weights_adv"\
    --summary "summary_adv" \
    --num_epoch 15
