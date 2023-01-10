python train_kpcn.py \
    --single_gpu \
    --batch_size 8 \
    --val_epoch 1 \
    --data_dir /home/kyubeom/ssd1/KPCN \
    --model_name KPCN_P_test \
    --desc "KPCN_P_test" \
    --num_epoch 10 \
    --lr_dncnn 1e-4 \
    --manif_loss FMSE \
    --pnet_out_size 12 \
    --lr_pnet 1e-4 \
    --use_llpm_buf \
    --manif_learn \
    --w_manif 0.1 \
    --device_id 3 \
    --save 'weights_full_4' \
    --summary 'summary_full_4' \
    --no_gbuf