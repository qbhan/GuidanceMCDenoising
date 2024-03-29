python train_adv.py \
    --single_gpu \
    --batch_size 8 \
    --val_epoch 1 \
    --data_dir /home/kyubeom/ssd1/KPCN \
    --model_name ADV_G \
    --desc "ADV_G" \
    --num_epoch 10 \
    --lr_G_diffuse 1e-4 \
    --lr_G_specular 1e-4 \
    --lr_D 1e-4 \
    --device_id 2 \
    --save "weights_adv"\
    --summary "summary_adv" \
    --num_epoch 15 \
    --decay_step 2