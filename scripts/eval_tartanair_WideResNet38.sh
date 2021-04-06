#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
PYTHONPATH=$PWD:$PYTHONPATH python3 eval_multi.py \
    --dataset tartanair_trav \
    --exp tartanair_multi_trav \
    --arch network.deepv3.DeepWV3Plus \
    --split trainval \
    --inference_mode sliding \
    --cv_split 2 \
    --scales 1.0 \
    --dump_images \
    --snapshot ${1} \
    --ckpt_path ${2}
