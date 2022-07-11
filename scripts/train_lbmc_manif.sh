python train_lbmc.py \
    --single_gpu \
    --batch_size 8 \
    --val_epoch 1 \
    --data_dir /mnt/ssd2/kbhan/SBMC \
    --model_name LBMC_manif_lrpnet5e5_w0.1 \
    --desc "LBMC_manif_lrpnet5e5_w0.1" \
    --num_epoch 10 \
    --lr_dncnn 1e-4 \
    --use_llpm_buf \
    --lr_pnet 5e-5 \
    --pnet_out_size 6 \
    --manif_learn \
    --manif_loss FMSE \
    --w_manif 0.1 \
    --device_id 2