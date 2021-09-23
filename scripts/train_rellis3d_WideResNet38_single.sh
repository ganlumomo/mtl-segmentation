#!/usr/bin/env bash

    # training RELLIS-3D
     python -m torch.distributed.launch --nproc_per_node=2 train.py \
        --dataset rellis3d \
        --cv 2 \
        --arch network.deepv3.DeepWV3Plus \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 300 \
        --lr 0.001 \
        --lr_schedule poly \
        --poly_exp 1.0 \
        --syncbn \
        --sgd \
        --crop_size 480 \
        --scale_min 1.0 \
        --scale_max 2.0 \
        --color_aug 0.25 \
        --max_epoch 90 \
        --img_wt_loss \
        --wt_bound 1.0 \
        --bs_mult 4 \
        --apex \
        --exp rellis_semantic_crop480 \
        --ckpt ./logs/ \
        --tb_path ./logs/


        # --snapshot ./pretrained_models/cityscapes_best.pth \
